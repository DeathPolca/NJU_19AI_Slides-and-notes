%(a)
num_con=conv([1 0.01],[1 5.5]);
den_con=conv([1 6.5],[1 0.0001]);
sys_con=tf(num_con,den_con);

num_func=[10];
den_func=[1 15 50 0];
sys_func=tf(num_func,den_func);

rlocus(sys_con*sys_func);
% K=8.58

%(b)
K=8.58;
sys_con1=tf(K*num_con,den_con);
sys_all=feedback(series(sys_func,sys_con1),[1]);
subplot(121)
step(sys_all);
% P.O.=1.77%,Ts=1.51s,

%(c)
Kv=10*8.58*0.01*5.5/10/6.5/0.0001/5;
sys_td=feedback(sys_func,sys_con1);
subplot(122)
step(sys_td)
%Kv=145.2意味着e_ss=1/Kv=0.0069