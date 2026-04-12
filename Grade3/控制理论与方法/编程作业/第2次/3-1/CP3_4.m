% (a)
A=[0 1 0; 0 0 1; -3 -2 -5];
B=[0;0;1];
C=[1 0 0];
D=[0];
sys_ss=ss(A,B,C,D);
sys=tf(sys_ss);%sys=1/s^3+5s^2+2s+3

% (b)
x0=[0 -1 1];
t=[0:0.01:10];
u=0*t;
[y,t,x]=lsim(sys_ss,u,t,x0);
plot(t,x(:,1),t,x(:,2),':',t,x(:,3),'--');
xlabel('t/s'),ylabel('x(t)'),grid
x_lsim=x(length(t),:)';

% (c)
dt=10;
Phi=expm(A*dt);
x_phi=Phi*x0'
%-0.2545 0.0418 0.1500
