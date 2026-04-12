%(a)
A=[0 1 0;0 0 1;-4.3 -1.7 -6.7];
B=[0;0;0.35];
C=[0 1 0];
D=[0];
p_con=[-1.4+1.4*j;-1.4-1.4*j;-2];
K=acker(A,B,p_con);
p_obv=[-18+5*j;-18-5*j;-20];
L=(acker(A',C',p_obv))'

%(b)
A1=[A -B*K;L*C A-B*K-L*C];
B1=[zeros(6,1)];
C1=eye(6);
D1=zeros(6,1);
sys_ss=ss(A1,B1,C1,D1);

%(c)
t=[0:0.001:10];
x0=[1;0;0;0.5;0.1;0.1];
[y,t]=initial(sys_ss,x0,t);
subplot(311)
plot(t,y(:,1),t,y(:,4),':')
subplot(312)
plot(t,y(:,2),t,y(:,5),':')
subplot(313)
plot(t,y(:,3),t,y(:,6),':')
