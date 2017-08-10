//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

//vocab_word结构体定义: cn为词频，word为词，code为huffman编码，codelen对应长度，point？
struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
//syn0对应为词向量矩阵。syn1对应为中间节点权重(hierachical softmax)
//syn1neg对应为负采样中间节点的权重
//expTable为计算exp(x)的快查表
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

//初始化以词频为参考的采样表table. 
//例如 词表：词频为{a:10, b:10, c:20} table_size=4;
//得到 table[1]=a_index table[2]=b_index table[3:4]=c_index
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  //总词频的权重计算，并不直接采用词频，而进行指数计算处理
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  //根据当前词的词频占比，划分table的分段，每一个分段段按词频比例存储该词下标。
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
//从文件指针指向位置，读取一个词，以空格，TAB, EOL作为分隔符
//如果是计算词表时调用，词表文件来自于训练过程中保存的格式为 '%s %lld\n' % (字符串，词频).读取前面%s部分
//如果是训练时调用，来自训练语料格式为 "%s %s ... %s\n"
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    //去除回车，空格，TAB, 换行符号
    if (ch == 13) continue;//字符 13 在ascii码表为回车键CR
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      //遇到分隔符挑出循环。"\n"额外进行回退操作
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin); 
        break;
      }
      //如果开始读入遇到换行符，返回"</s>"标记。
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    //读取字符
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   //长度超过21的字符进行截断,对于汉字相当于设定最长为10个字。
  }
  word[a] = 0;
}

// Returns hash value of a word
//哈希算法为 for(char in word) hash+=hash*257+char
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
// 查找vocab_hash，返回当前词在词表中下标。
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
//读入一个词，返回其在词表vocab的下标
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
// 在词表vocab中增加一个词，词频初始为0。
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  // 词表不够，通过realloc追加内存段
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  //将词hash，存入hash表，如果发生冲突,则顺延一个位置
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  //hash表中存储词的下标
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
// 根据词频进行词的比较
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
// 根据词频排序词表vocab,重新初始化并计算vocab_hash
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  // </s>放在第一位，其余根据词频排序
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // 词频低于 min_count[default:5]抛弃
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      //累积训练的词频计数
      train_words += vocab[a].cn;
    }
  }
  //根据前面的删减，重新安排内存。
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // 为huffman_code和point分配内存, 建树做准备
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
// 以min_reduce[default:1]去除低频词，缩减词表。
// 调用ReduceVocab()的前置条件是当前词表总词数到达vocab_hash_size的70%
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
// 根据词频建huffman树
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  //临时数组，用于记录某节点和临时节点的频次
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  //临时数据，用于记录某节点的d_label
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  //临时数组，记录某节点的父节点位置
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // 建树，每次只生成一颗节点
  for (a = 0; a < vocab_size - 1; a++) {
    // 从最后一个词开始，找到词频最小的两个节点{min1,min2}合成一棵树，其词频和和存在父节点节点
    // 其中新生成的父节点 由pos2指向，会参与min1, min2的对比选拔
    // 令parent_note中指向该父节点
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    // 将min2i{即右节点}的d_label设为1
    // min1i{即左节点}的d_label如初始化设定，为0
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  // 从树的根节点位置到词的路径上的d_label赋值给词的code,记录在对应vocab[a].code
  // 路径从叶结点向根节点遍历赋值给词的point数组。再按自顶向下的顺序,记录在对应vocab[a].point数组
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b; //point父节点列表
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;              // point[0]指向停止条件
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];            // 存储路径的编码例如0001
      vocab[a].point[i - b] = point[b] - vocab_size; // 指向从根节点往下到目标词的路径
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

//从语料中统计出词频和词表
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  //初始化vocab_hash表
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  //先将"</s>"加入词表
  AddWordToVocab((char *)"</s>");
  while (1) {
    //从训练语料中读取词
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    //每处理10w词，以fflush方式输出一次log
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    //取得当前词word在词表中的下标，其中-1表示无当前词
    i = SearchVocab(word);
    if (i == -1) {
      //不在词表则加入，并计数为1
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    //词表过大，进行缩减
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  //根据词频排序词表vocab,重新初始化并设置vocab_hash
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  //得到文件总大小file_size
  file_size = ftell(fin);
  fclose(fin);
}

//按 "%s %lld\n" % (词，词频)格式输出
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

//读取来自Vocab的词表，训练参数中词表给定的情况下会使用本函数
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  //初始化vocab_hash，size[default:30,000,000]，用于判断词是否在词表内。
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    //读取一个字, '<\s>'为词表第一项
    ReadWord(word, fin);
    if (feof(fin)) break;
    //将词加入词表vocab，同时计入vocab_hash表
    a = AddWordToVocab(word);
    //读取预留词频
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  //读取训练集大小,得到file_size
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  // syn0分配一段对齐的内存. 大小为|词表*词向量维度|的浮点型内存
  // syn0为词向量参数矩阵
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real)); // real为float
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  //如果是hierarchical softmax
  if (hs) {
    // syn1分配一段对齐的内存. 大小为|词表*词向量维度|的浮点型内存
    // syn1为weight矩阵
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    //初始化syn1
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  //如果是negative sampling策略
  if (negative>0) {
    // syn1neg分配一段对齐的内存. 大小为|词表*词向量维度|的浮点型内存
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    //初始化syn1neg
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  //随机初始化，syn0指向的内存
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree();
}

//词向量训练函数
void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  //neu1 对应窗口内上下文的累加
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  //neu1e 对应词向量上更新量 
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  //读入训练文件
  FILE *fi = fopen(train_file, "rb");
  //每个线程负责一部分语料的计算
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
      // 每增加10000字，alpha按词频衰减一次，直到starting_alpha * 0.0001
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      // 最大读入1k个字, 换行会'\n'替换为'</s>'符号。
      while (1) {
        // 读入，word为词在vocab中下标
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        // 以一定概率跳过当前词, 概率由词频计算得到
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        // 将未被跳过的词组合成为句子
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    // 读完线程所属于数据区(未严格按文件区域大小，以平均分配词数的简单策略进行数据区大小控制)
    // 重新开始新的一轮迭代
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    word = sen[sentence_position];
    if (word == -1) continue;
    // 上下文累加和neu1
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    // 权重的更新量
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    // 随机取窗口window(=5)内的数b
    // 相当于在sentence_position的左右邻域[-(windows - b), windows - b]中取值
    b = next_random % window;
    if (cbow) {  // train the cbow architecture
      // in -> hidden
      cw = 0;
      // 再进行子窗口b取词
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        //累加上下文的各词向量
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
        //上下文词向量求平均
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        //如果是hierachical softmax
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // 前馈将上下文词向量从隐层沿着树的根节点往下传, syn1为中间节点的参数
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 计算梯度。g 为梯度，alpha为学习率, f为logistic function输出的预测值
          g = (1 - vocab[word].code[d] - f) * alpha;
          // 累积中间节点的参数和梯度, 计算词向量的更新量：delta = delta + g * w(cur)
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // 更新中间节点的参数: w(cur) = w(cur) + g * v(mid)
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        //目标函数为NEGATIVE SAMPLING,负采样无树结构
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          //生成正样本
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            //生成负样本, 从table中取得采样
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          //前馈将上下文词向量从隐层往输出层传, syn1neg为每个词对应的参数
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          //计算词向量的更新量：累积各样本参数和梯度 
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          //更新负采样中采样点的weight参数 
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        //梯度由隐层更新到输入层,对每一个上下文更新梯度
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } else {  
      //skip-gram方式训练
      //窗口内再进行随机取上一个子窗口 size=b
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // 前馈将上下文词向量从隐层沿着树的根节点往下传, syn1为中间节点的参数
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 计算梯度。g 为梯度，alpha为学习率, f为logistic function输出的预测值
          g = (1 - vocab[word].code[d] - f) * alpha;
          // 计算词向量的更新量：累积中间节点的参数和梯度
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // 更新中间节点的weight参数
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        //目标函数为NEGATIVE SAMPLING,负采样无树结构
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          //生成正样本
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            //生成负样本, 从table中取得采样
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          //前馈将上下文词向量从隐层往输出层传, syn1neg为每个词对应的参数
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          //计算词向量的更新量：累积各样本参数和梯度 
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          //更新对应节点的weight参数 
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        /// 更新词向量
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  //多线程数目，来自参数num_threads
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  //词表可配置，来自参数-read-vocab，也可根据输入语料统计得到
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  //词表可配置，来自参数-save-vocab
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  //建树操作，以及为negative sampling分配辅助内存
  InitNet();
  //为negative sampling初始化采样表
  if (negative > 0) InitUnigramTable();
  start = clock();
  //多线程调用词向量训练函数
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  //保存词向量,以二进制或者明文方式输出到文件中
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // 设置了classes参数，执行K-means聚类
    int clcn = classes, iter = 10, closeid;
    // 每个类别，
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    //初始化词属于某个类别
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    //迭代更新cluster
    for (a = 0; a < iter; a++) {
      //初始化中心向量
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      //每个中心包含的词个数
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      //累积每个类所有的词向量到类的中心向量中
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      //更新类的中心向量
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      //更新每个词的类别归属
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    //保存每个词及其对应类别 
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  //-size 传入词向量的大小
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  //在将[-6, 6]区间，预分为1000个子区间。
  //计算好每个区间的logistic function值并存储在expTable表中。方便直接查询
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() function;
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1), 即f(x) = 1/(1+x^{-1});
  }
  TrainModel();
  return 0;
}
