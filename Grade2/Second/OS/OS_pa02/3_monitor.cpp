
#include <iostream>
#include <queue>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
 
using namespace std;
class Queue{
private:
  queue<int> _store; //初始化一个存放数据的队列
  int _capacity;//队列的总容量
  pthread_mutex_t mutex;//互斥锁
  pthread_cond_t  cond_Producer;
  pthread_cond_t  cond_Consumer;
 
public:
  Queue(int capacity = 5)//构造函数 将队列的总容量设为5
    :_capacity(capacity)
  {
    pthread_mutex_init(&mutex, NULL);//初始化互斥锁
    pthread_cond_init(&cond_Consumer, NULL);
    pthread_cond_init(&cond_Producer, NULL);
  }
  bool push(const int& data)
  {
      pthread_mutex_lock(&mutex);//上锁
      while(_capacity ==_store.size())//判断当前队列是否已满
      {//如果队列满，则
        pthread_cond_signal(&cond_Consumer);
        pthread_cond_wait(&cond_Producer,&mutex);
      }
      //如果队列不满，则
      _store.push(data);//插入数据
      pthread_cond_signal(&cond_Consumer);
      pthread_mutex_unlock(&mutex);//解锁
    return true;
  }
 
  bool pop(int& data)
  {
    pthread_mutex_lock(&mutex);//上锁
    while(_store.empty())//判断队列是否为空
    {//如果队列为空，则
      pthread_cond_signal(&cond_Producer);
      pthread_cond_wait(&cond_Consumer,&mutex);
    }
    //如果队列不为空，则
    data = _store.front();//从队列中取出一个元素，这里取出的是当前队列的第一个元素
    _store.pop();//pop操作
    pthread_cond_signal(&cond_Producer);
    pthread_mutex_unlock(&mutex);//解锁
    return true;
  }
};
 
void* producer(void* arg){
  srand((unsigned) time(NULL)); //随机种子函数
  Queue* p =  (Queue*)arg;
  while(1){
    int data = rand()%11;
    p->push(data);
  }
  return NULL;
}
 
void* consumer(void* arg){
  Queue* p = (Queue*)arg;
  int data = 0;
  while(1)
  {//消费者消费一个数据
    p->pop(data);
  }
  return NULL;
}
 
 
int main()
{
  pthread_t Consumer;
  pthread_t Producer;
  Queue arr; 
  pthread_create(&Consumer, NULL, consumer,(void*)&arr);
  pthread_create(&Producer, NULL, producer,(void*)&arr);
  pthread_join(Consumer,NULL);
  pthread_join(Producer,NULL);
  return 0;
}
