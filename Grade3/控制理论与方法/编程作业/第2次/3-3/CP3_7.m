A=[0 1;-2 -3];
B=[0;1];
C=[1 0];
D=[0];
sys_tf=tf(ss(A,B,C,D));
x0=[1 0];
t=[0:0.01:10];
u=0*t;
[y,t,x]=lsim(ss(A,B,C,D),u,t,x0);
plot(t,x(:,1),t,x(:,2),':');
legend("x1","x2")
xlabel('t/s'),ylabel('State response'),grid