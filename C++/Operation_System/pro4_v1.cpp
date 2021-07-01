// Jian Sun Project 4 Reader-Writer Problem

#include <iostream>
#include <cmath> 
#include <array>
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <algorithm>
#include <semaphore.h>
using namespace std;

// shared variable
int LOAD_NUM = 20;
int READ_NUM = 6;       // read thread number
int WRITE_NUM = 3;	    // write thread number
int RC = 0;
const int N = 3;

// mutex to protect shared data (critical region)
pthread_mutex_t rd_lock;
pthread_mutex_t wt_lock;

void *reader(void *arg);
void *writer(void *arg);

int main(int argc, char* argv[])
{
	pthread_t rid[READ_NUM];
	pthread_t wid[WRITE_NUM];
	int rarg [READ_NUM];
	int warg [WRITE_NUM];

	pthread_mutex_init(&rd_lock, NULL);
	pthread_mutex_init(&wt_lock, NULL);
	srand(time(NULL));
 	
 	for (int i=0; i<READ_NUM; i++){
 		rarg[i]=i;
 		int ST_READ = pthread_create(&rid[i], NULL, &reader, &rarg[i]);    
        if (ST_READ != 0) cerr << "Unable to create thread" << endl;
    }

 	for (int j=0; j<WRITE_NUM; j++){
 		warg[j]=j;
 		int ST_WRITE = pthread_create(&wid[j], NULL, &writer, &warg[j]);
 		if (ST_WRITE != 0) cerr << "Unable to create thread" << endl;
    }

 	for (int i=0; i<READ_NUM; i++){
        pthread_join(rid[i], NULL);
    }
    for (int j=0; j<WRITE_NUM; j++){ 
    	pthread_join(wid[j], NULL);
    } 
}

void *reader(void *arg) {
	int RT_ID = *static_cast<int*>(arg);
	for(int rtime=0; rtime<N; rtime++){
		pthread_mutex_lock(&rd_lock);
		RC=RC+1;
		if (RC==1) pthread_mutex_lock(&wt_lock);
		pthread_mutex_unlock(&rd_lock);
		cout<<"reader "<<RT_ID<<" read "<<LOAD_NUM<<" "<<RC<<" reader(s)"<<endl;
		pthread_mutex_lock(&rd_lock);
		RC=RC-1;
		if (RC==0) pthread_mutex_unlock(&wt_lock);
		pthread_mutex_unlock(&rd_lock);     
		pthread_exit(0);   
		usleep((rand() % 50) + 10);
	}
}
void *writer(void *arg) {
	int WT_ID = *static_cast<int*>(arg);
	for(int wtime=0; wtime<N; wtime++) {
		pthread_mutex_lock(&wt_lock);
		LOAD_NUM = rand();
		cout<<"writer "<<WT_ID<<" write "<<LOAD_NUM<<" "<<RC<<" reader(s)"<<endl;
		pthread_mutex_unlock(&wt_lock);	
		pthread_exit(0); 
		usleep((rand() % 50) + 10);
	} 
}
