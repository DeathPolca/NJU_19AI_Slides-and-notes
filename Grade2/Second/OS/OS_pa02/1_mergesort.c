#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <malloc.h>
int n;
int a[1000000];
 
void merge(int a[],int low,int mid,int high)
{
	int i,k;
	int* temp = (int*)malloc((high-low+1)*sizeof(int));//合并空间
	int start1 = low;
	int end1 = mid;
	int start2 = mid+1;
	int end2 = high;
	//比较两个指针所指向的元素，选择相对小的元素放入到合并空间，并移动指针到下一位置
	for (k = 0;start1<=end1 && start2<=end2;k++)
	{
		if (a[start1]<a[start2])
			temp[k] = a[start1++];
		else
			temp[k] = a[start2++];
	}
	//检测剩余项，若有剩余，直接拷贝出来粘到合并序列尾
	while (start1<=end1)
		temp[k++] = a[start1++];
	while (start2<=end2)
		temp[k++] = a[start2++];
	//将排好序的数组拷贝到原数组
	for (i = 0;i<high-low+1;i++)
		a[low+i] = temp[i];
	//合并过程
	printf("%d:",mid);
	for (i = 0;i<high-low+1;i++)
		printf("%d ",a[low+i]);
	printf("\n");
	free(temp);
}
void merge_sort(int a[],int low,int high)
{
	int mid = (low+high)/2;
	if (low <high)
	{
		merge_sort(a,low,mid);
		merge_sort(a,mid+1,high);
		merge(a,low,mid,high);
	}
}
void thread_function ()
{
    merge_sort(a,0,n-1);
    for(int i=0;i<n;i++){
        printf("%d",a[i]);
    }
}
int main(){
	int temp;
    scanf("%d",&temp);
	n=temp;
	int a1[n];
    for(int i=0;i<n;i++){
        scanf("%d",&a1[i]);
		a[i]=a1[i];
    }
    pthread_t *thrd = (pthread_t *) malloc(1*sizeof(pthread_t));
    void *pi_exit_status;
    pthread_create(&thrd[0], NULL, thread_function, NULL);
    pthread_join(thrd[0], &pi_exit_status);

    return 0;
}