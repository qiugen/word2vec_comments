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

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char *bestw[N];
  char file_name[max_size], st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, cn, bi[100];
  char ch;
  float *M;
  char *vocab;
  if (argc < 3) {
    printf("Usage: ./distance <BIN_FILE> <OUT_FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  //for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    //for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    //len = 0;
    //for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    //len = sqrt(len);
    //for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);

  printf("File out: %s\n", argv[2]);
  fstream fout(argv[2], ios::out);
  //fout << words << " " << size << "\t#File: " << file_name << endl;
  printf("%lld %lld #File: %s\n",words,size,file_name);
  for(a = 0; a < words; a++)
  {
   //   printf("%s ", &vocab[a * max_w]);
      fout << &vocab[a * max_w] << std::endl;
      if (a>10){break;}
      //for(b = 0; b < size; b++)
      //{
      //    fout << M[a*size + b] << " ";
//    //      printf("%0.4f ", M[a * size + b]);
      //}
      //fout << endl;
//      printf("\b\b\n");
  }
  fout.close();
  return 0;
}
