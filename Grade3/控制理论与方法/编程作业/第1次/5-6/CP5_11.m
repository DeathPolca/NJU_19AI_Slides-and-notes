%(a)
num1=[0.5 2]
den1=[1 0]
sys1=tf(num1,den1);
num2=[1];
den2=[1 2 0];
sys2=tf(num2,den2);
sys3=series(sys1,sys2);%上面的串联
sys=feedback(sys3,[1]);%加反馈

%(b)
t=[0:0.05:50];
subplot(311)
impulse(sys,t);
subplot(312)
step(sys,t);
subplot(313)
u=t;
lsim(sys,u,t);
