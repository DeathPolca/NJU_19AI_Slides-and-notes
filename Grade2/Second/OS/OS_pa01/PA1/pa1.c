#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <malloc.h>
#include <time.h>
void *calthd(void *arg){//calculate thread
    /*int t = *(int *) arg;
    int x,y;
    int i = 0;
    int *count =0;
    for(i = 0;i<t;i++){
        float x = rand()/(RAND_MAX+1.0);
        float y = rand()/(RAND_MAX+1.0);
        if(x*x+y*y<=1) count++;
    }
    return (int *)1;*/
    int t = *(int *) arg;
    int i = 0;
    int val = *(int *) arg;
    int *res = (int *) malloc(sizeof(int));     // 为计算结果的存储分配一个空间
    *res = 0;
    srand(time(NULL));
    //printf("%s\n",(char *)res);
    for(i = 0;i<t;i++){
        double x = (double)rand()/RAND_MAX;
        double y = (double)rand()/RAND_MAX;
        //printf("%f-%f\n",x,y);
        if(x*x+y*y<=1){
            *res=*res+1;
        } 
    }
    //printf("%d\n",*(int *)res);
    return res;

}
int main(){
    int num;//maximum 10
    scanf("%d",&num);
    clock_t start = clock();
    pthread_t *thrd = (pthread_t *) malloc(num * sizeof(pthread_t));
    int t = 1000000/num;
    int i = 0;
    int *arr = (int *) malloc(num * sizeof(int));
    for(;i<num;i++){
        arr[i] = t;
        pthread_create(&thrd[i], NULL, calthd, &arr[i]);
    }
    int sum = 0;
    for(i = 0;i<num;i++){
        void *res;
        pthread_join(thrd[i], &res);
        //printf("%d\n", *(int *)res);
        sum += *(int *)res;
        /*free(res);
        res = NULL;*/
        //printf("\n");
    }
    /*free(thrd);
    free(arr);*/
    double r = 4*(double)sum/1000000;
    printf("%f\n",r);
    clock_t end = clock();
    double duration;
    duration=(double)(end)-(double)(start);
    printf("%f\n",duration);
    return 0;
}
