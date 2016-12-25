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
#include <string.h>
#include <math.h>
#include <malloc.h>

const long long max_size = 2000;         // max length of strings
const long long N = 20;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int recursive_search(char *candi, float* M, int words_len, char* vocab, int size, int round, int max_round){
    if (round == max_round) {
        return 0;
    }
    //当前的轮数
    round ++;
    //循环内使用的临时变量
    int a=0,b=0,c=0,d=0;
    long long bi=0;
    //最佳候选存储位置
    char *bestw[N];
    //dist cosin距离。len 临时存放向量的2范数。 
    //bestd最佳topn的词得分。vec存储词向量临时变量
    float dist, len, bestd[N], vec[max_size];
    //找到candi在词向量词典的下标
    for (b = 0; b < words_len; b++) if (!strcmp(&vocab[b * max_w], candi)) break;
    if (b == words_len) {
        b = -1;
    }
    bi = b;
    if (b == -1) {
        printf("Out of dictionary word!\n");
        return -1;
    }
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (a = 0; a < size; a++) vec[a] = 0;
    //初始化候选向量
    for (a = 0; a < size; a++) vec[a] += M[a + bi* size];
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    //初始化权重
    for (a = 0; a < N; a++) bestd[a] = -1;
    //初始化候选
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words_len; c++) {
      a = 0;
      if (bi == c) a = 1;
      if (a == 1) continue;
      dist = 0;
      //计算相似度
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);
          break;
        }
      }
    }

    for (a = 0; a < N; a++) printf("%s\t\t%f\n", bestw[a], bestd[a]);
    for (a = 0; a < N; a++) {
        recursive_search(bestw[a], M, words_len, vocab, size, round, max_round);
    }
    return 0;
}
//filename: 文件名
//ret_words_len: 词表总大小
//ret_size: 词向量大小
//ret_M: 词向量的值数组   
//ret_vocab: 明文词表
//注释： 为了返回指针,使用了二级指针
int load_bin_model(const char* file_name, long long* ret_words_len, long long* ret_size, float **ret_M, char **ret_vocab) {
  FILE *f; //bin文件指针
  float len; //临时变量存储二范数
  float *M=NULL;
  char *vocab=NULL;
  long long words_len, size, a, b;

  f = fopen(file_name, "rb");
  if (f == NULL) {
    fprintf(stderr, "Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words_len);
  fscanf(f, "%lld", &size);
  *ret_words_len = words_len;
  *ret_size = size;
  vocab = (char *)malloc((long long)words_len * max_w * sizeof(char));
  M = (float *)malloc((long long)words_len * (long long)size * sizeof(float));
  if (M == NULL) {
    fprintf(stderr, "Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)(words_len * size * sizeof(float) / 1048576), words_len, size);
    return -1;
  }
  //读取模型词典
  for (b = 0; b < words_len; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  *ret_M = M;
  *ret_vocab = vocab;
  fclose(f);
  return 0;
}

//filename: 文件名
//ret_words_len: 词表总大小
//ret_size: 词向量大小
//ret_M: 词向量的值数组   
//ret_vocab: 明文词表
//注释： 为了返回指针,使用了二级指针
int load_txt_model(const char* file_name, long long* ret_words_len, long long* ret_size, float **ret_M, char **ret_vocab) {
  FILE *f; //bin文件指针
  float len; //临时变量存储二范数
  float *M=NULL;
  char *vocab=NULL;
  long long words_len, size, a, b;

  f = fopen(file_name, "rb");
  if (f == NULL) {
    fprintf(stderr, "Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words_len);
  fscanf(f, "%lld", &size);
  *ret_words_len = words_len;
  *ret_size = size;
  vocab = (char *)malloc((long long)words_len * max_w * sizeof(char));
  M = (float *)malloc((long long)words_len * (long long)size * sizeof(float));
  if (M == NULL) {
    fprintf(stderr, "Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)(words_len * size * sizeof(float) / 1048576), words_len, size);
    return -1;
  }
  //读取模型词典
  for (b = 0; b < words_len; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fscanf(f, "%f", &M[a + b * size]);
        //fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  *ret_M = M;
  *ret_vocab = vocab;
  fclose(f);
  return 0;
}


int main(int argc, char **argv) {
  char candi[max_size]; //输入候选词
  char file_name[max_size];//候选词向量 
  long long words_len, size, a;//words_len
  float *M=NULL;
  char *vocab=NULL;
  int round=0;
  int eof = 0;
  //最大迭代次数
  int max_round = 1;
  if (argc < 2) {
    fprintf(stderr, "Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY(!) FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);

  int ret = load_bin_model(file_name, &words_len, &size, &M, &vocab);
  //int ret = load_txt_model(file_name, &words_len, &size, &M, &vocab);
  if (ret < 0) {
    fprintf(stderr, "load model failed");
    return -1;
  }
  else {
    fprintf(stderr, "load model successed, len %lld size %lld\n", words_len, size);
  }
  while (1) {
    printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    while (1) {
      candi[a] = fgetc(stdin);
      if (candi[a] == EOF) {
          eof = 1;
          break;
      }
      round = 0;
      if ((candi[a] == '\n') || (a >= max_size - 1)) {
        candi[a] = 0;
        break;
      }
      a++;
    }
    if (eof == 1) {
        break;
    }
    if (!strcmp(candi, "EXIT")) break;
    recursive_search(candi, M, words_len, vocab, size, round, max_round);
  }
  return 0;
}
