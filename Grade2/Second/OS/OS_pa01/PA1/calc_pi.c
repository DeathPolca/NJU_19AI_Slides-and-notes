#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <malloc.h>
#include <time.h>
void * thread_function (void *arg)
{
    int t = *(int *)arg;
    int k=RAND_MAX;
    int *count = (int *) malloc(sizeof(int));     // 为计算结果的存储分配一个空间
    *count = 0;
    srand(time(NULL));
    double x, y;
    for(int i = 0;i<t;i++){
        x = (double)rand()/k;
        y = (double)rand()/k;
        if(x*x+y*y<=1){
            *count=*count+1;
        } 
    }
    return count;
}
int main(){
    int num;//同时工作的线程数
    int sum=0;
    scanf("%d",&num);
    pthread_t *thrd = (pthread_t *) malloc(num*sizeof(pthread_t));
    int t = 1000000/num;//每个线程的采样次数
    clock_t start = clock();    
    for(int i=0;i<num;i++){
        pthread_create(&thrd[i], NULL, thread_function, &t);
    }
    for(int i = 0;i<num;i++){
        void *count;
        pthread_join(thrd[i], &count);
        sum += *(int *)count;
    }
    clock_t end = clock();    
    double pi = 4*(double)sum/1000000;
    printf("%f\n",pi);
    double cost;
    cost=end-start;
    printf("the cost is %f\n",cost);
    return 0;
}