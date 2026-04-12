num=15;
den=[1 8 15];
t=[0:0.05:3];
%用impulse
sys=tf(num,den);
y1=impulse(sys,t);
%解析方法
y2=7.5*exp(-3.*t)-7.5*exp(-5.*t);
plot(t,y1,'r',t,y2,'c*');
legend('impulse','解析方法')