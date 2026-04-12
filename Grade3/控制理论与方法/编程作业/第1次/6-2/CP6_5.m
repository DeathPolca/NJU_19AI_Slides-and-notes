tau1=2;
tau2=0.5;
K=1;
%反应较快
tau=0.1;
num1=[-K*tau1*tau -K*tau+2*K*tau1 2*K];
den1=[tau2*tau 2*tau2+tau 2];
sys1=tf(num1,den1);

num2=[-10];
den2=[1 10];
sys2=tf(num2,den2);

num3=[-1 -6];
den3=[1 3 6 0];
sys3=tf(num3,den3);

syss1=series(sys1,sys2);%前两个串联
syss2=series(syss1,sys3);%和第三个串联
sys=feedback(syss2,[1]);%单位反馈
p_fast=pole(sys);%反应快的极点
%反应较慢
tau=0.6;
num1=[-K*tau1*tau -K*tau+2*K*tau1 2*K];
den1=[tau2*tau 2*tau2+tau 2];
sys1=tf(num1,den1);

num2=[-10];
den2=[1 10];
sys2=tf(num2,den2);

num3=[-1 -6];
den3=[1 3 6 0];
sys3=tf(num3,den3);

syss1=series(sys1,sys2);%前两个串联
syss2=series(syss1,sys3);%和第三个串联
sys=feedback(syss2,[1]);%单位反馈
p_slow=pole(sys);%反应慢的极点

%最大允许时延，用劳斯判据法可求tau=0.2045
tau=0.2045;
num1=[-K*tau1*tau -K*tau+2*K*tau1 2*K];
den1=[tau2*tau 2*tau2+tau 2];
sys1=tf(num1,den1);

num2=[-10];
den2=[1 10];
sys2=tf(num2,den2);

num3=[-1 -6];
den3=[1 3 6 0];
sys3=tf(num3,den3);

syss1=series(sys1,sys2);%前两个串联
syss2=series(syss1,sys3);%和第三个串联
sys=feedback(syss2,[1]);%单位反馈
p=pole(sys);