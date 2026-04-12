num=[25];
den=[1 5 0];
sys1=tf(num,den);
sys=feedback(sys1,[1]);%整个系统
step(sys);
hold on
%Ts
X=1.62;
Y=0.98;
plot(X,Y,'ko','markerfacecolor','k')
%Tp
X=0.718;
Y=1.16;
plot(X,Y,'ko','markerfacecolor','k')