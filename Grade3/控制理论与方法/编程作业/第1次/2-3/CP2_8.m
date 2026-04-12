den=[1 3 20 ]
%z=5
num1=[4 20];
sys1=tf(num1,den);
%z=10
num2=[2 20];
sys2=tf(num2,den);
%z=15
num3=[20/15 20];
sys3=tf(num3,den);
%drawing
t=[0:0.005:5];
[y1,t1]=step(sys1,t);
[y2,t2]=step(sys2,t);
[y3,t3]=step(sys3,t);
plot(t1,y1,'r'),grid
hold on
plot(t2,y2,'g'),grid
hold on
plot(t3,y3,'b'),grid
legend('z=5','z=10','z=15')
xlabel("time")
ylabel("响应")


