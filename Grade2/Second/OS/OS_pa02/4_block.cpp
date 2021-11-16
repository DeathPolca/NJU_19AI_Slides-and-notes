
#include <iostream>
#include <thread>
#include <mutex>
using namespace std;
mutex source1;
mutex source2;
 
void threadT1()
{
    unique_lock<mutex> lockA(source1);
	//cout << "threasT1 got source1" << endl;
	// 线程1睡眠2s再获取source2，保证source2先被线程2获取，模拟死锁问题的发生
	this_thread::sleep_for(chrono::seconds(2));
	unique_lock<mutex> lockB(source2);
}
 
 
void threadT2()
{
	unique_lock<mutex> lockB(source2);
	//cout << "threasT2 got source2" << endl;
	// 线程2睡眠2s再获取source1，保证source1先被线程1获取，模拟死锁问题的发生
	this_thread::sleep_for(chrono::seconds(2));
	unique_lock<mutex> lockA(source1);
}
 
int main()
{
	thread t1(threadT1);
	thread t2(threadT2);
	// main主线程等待所有子线程执行完
	t1.join();
	t2.join();
	return 0;
}
