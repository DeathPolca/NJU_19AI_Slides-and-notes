t=[0:0.01:30];
%(1)
num=[4];
den=[1 0 4];
sys1=tf(num,den);
y1=impulse(sys1,t);
%(2)
num=[4];
den=[1 0.4 4];
sys2=tf(num,den);
y2=impulse(sys2,t);
%(3)
num=[1];
den=[1 0 1];
sys3=tf(num,den);
y3=impulse(sys3,t);
%(4)
num=[1];
den=[1 0.4 1];
sys4=tf(num,den);
y4=impulse(sys4,t);

clf
subplot(221),plot(t,y1),title('(1)')
subplot(222),plot(t,y2),title('(2)')
subplot(223),plot(t,y3),title('(3)')
subplot(224),plot(t,y4),title('(4)')