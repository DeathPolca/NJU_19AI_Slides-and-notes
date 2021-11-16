# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <sys/types.h>
# include <pthread.h>
# include <semaphore.h>
# include <string.h>

// 定义传入线程执行函数的参数
struct p {
    int id;
    int runtime;
    int sleeptime;
};

// 信号量的定义
sem_t readcount, wrt, mutex;

// 写者定义
void* writer(void* param) {
    struct p data = *(struct p*)param;
    sleep(data.sleeptime);
    sem_wait(&wrt);
    printf("Process %d, writing!\n", data.id);
    sleep(data.runtime);
    sem_post(&wrt);
    pthread_exit(0);
}

// 读者定义
void* reader(void* param) {
    struct p data = *(struct p*)param;
    sleep(data.sleeptime);
    sem_wait(&mutex);
    //readcount++;
    int value;
    sem_getvalue(&readcount, &value);
    sem_init(&readcount, 0, value+1);
    sem_getvalue(&readcount, &value);
    if (value == 1)
        sem_wait(&wrt);
    sem_post(&mutex);
    printf("Process %d, reading!\n", data.id);
    sleep(data.runtime);
    sem_wait(&mutex);
    //readcount--;
    sem_getvalue(&readcount, &value);
    sem_init(&readcount, 0, value-1);
    sem_getvalue(&readcount, &value);
    if (value == 0)
        sem_post(&wrt);
    sem_post(&mutex);
    pthread_exit(0);
}

int main() {
    int Thread_num, i;
    struct p* data;
    pthread_t* pthreads;
    pthread_attr_t* attr;
    sem_init(&mutex, 0, 1);
    sem_init(&readcount, 0, 0);
    sem_init(&wrt, 0, 1);
    data = (struct p*)malloc((Thread_num+1)*sizeof(struct p));
    pthreads = (pthread_t*)malloc((Thread_num+1)*sizeof(pthread_t));
    attr = (pthread_attr_t*)malloc((Thread_num+1)*sizeof(pthread_attr_t));
    for (i = 0; i < Thread_num; i++) {
        pthread_attr_init(&attr[i]);
        scanf("%d", &data[i].id);
        getchar();
        char C;
        scanf("%c", &C);
        if (C == 'R') {
            scanf("%d", &(data[i].sleeptime));
            scanf("%d", &(data[i].runtime));
            pthread_create(&pthreads[i], &attr[i], reader, &data[i]);
        } else if (C == 'W') {
            scanf("%d", &(data[i].sleeptime));
            scanf("%d", &(data[i].runtime));
            pthread_create(&pthreads[i], &attr[i], writer, &data[i]);
        }
    }
    sem_destroy(&mutex);
    sem_destroy(&readcount);
    sem_destroy(&wrt);
    return 0;
}