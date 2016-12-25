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
#include <fstream>

using namespace std;

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries


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
  FILE *fo;
  char file_name[max_size];
  long long words, size, a, b; 
  float *M;
  char *vocab;
  if (argc < 3) {
    printf("Usage: ./distance <BIN_FILE> <OUT_FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);


  int ret = load_txt_model(file_name, &words, &size, &M, &vocab);
  if (ret < 0) {
    fprintf(stderr, "load txt model failed");
    return -1;
  }
  printf("File out: %s\n", argv[2]);
  //fstream fout(argv[2], ios::out);
  fo = fopen(argv[2], "wb");
  //fout << words << " " << size << "\t#File: " << file_name << endl;
  fprintf(fo, "%lld %lld\n",words,size);
  printf("wordslen:%lld, size:%lld", words, size);
  for(a = 0; a < words; a++)
  {
      fprintf(fo, "%s ", &vocab[a * max_w]);
      //fout << &vocab[a * max_w] << " ";
      for(b = 0; b < size; b++)
      {
            //fout << M[a*size + b] << " ";
            fwrite(&M[a*size + b], sizeof(float), 1, fo);
//          printf("%0.4f ", M[a * size + b]);
      }
      //fout << endl;
      fprintf(fo, "\n");
//      printf("\b\b\n");
  }
  //fout.close();
  fclose(fo);
  free(M);
  free(vocab);
  return 0;
}
