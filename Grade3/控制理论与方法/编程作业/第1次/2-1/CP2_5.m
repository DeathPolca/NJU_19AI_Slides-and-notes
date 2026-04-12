%(a)
a=1;
b=8;
k=10.8e+08;
j=10.8e+08;
num=k*[1 a];
den=j*[1 b 0 0];
sys=tf(num,den);
sys_all=feedback(sys,[1]);%把反馈部分带上

%(b)
t=[0:0.005:80];
f=10*pi/180;%10°
ret=sys_all*f;
y=step(ret,t);
plot(t,y*180/pi,'r'),grid
hold on

%(c)
%降到80%
j=10.8e+08*0.8;
den=j*[1 b 0 0];
sys=tf(num,den);
sys_all=feedback(sys,[1]);
ret=sys_all*f;
y=step(ret,t);
plot(t,y*180/pi,'g'),grid
hold on
%降到50%
j=10.8e+08*0.5;
den=j*[1 b 0 0];
sys=tf(num,den);
sys_all=feedback(sys,[1]);
ret=sys_all*f;
y=step(ret,t);
plot(t,y*180/pi,'b'),grid
legend('j','0.8j','0.5j');
xlabel('time')
ylabel('响应')