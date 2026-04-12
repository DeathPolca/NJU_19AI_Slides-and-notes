num=[1 10];
den=[1 15 0 0];
t=[0:0.1:50];
sys1=tf(num,den);%前向通路
sys=feedback(sys1,[1]);%整体传递函数
u=t;
lsim(sys,u,t);
%看图像知道稳态误差为0