%(a)
A=[0 0 0 1 0 0;0 0 0 0 1 0;0 0 0 0 0 1;7.3809 0 0 0 2 0;0 -2.1904 0 -2 0 0;0 0 -3.1904 0 0 0];
B1=[0;0;0;1;0;0];
B2=[0;0;0;0;1;0];
B3=[0;0;0;0;0;1];
C=[0 1 0 0 0 0];
D=[0];
sys_ss1=ss(A,B1,C,D);
sys_ss2=ss(A,B2,C,D);
sys_ss3=ss(A,B3,C,D);
eva=eig(A)
%不稳定，虚轴和右半平面上有极点

%(b)
ctr_1=ctrb(sys_ss1);
det1=det(ctr_1)
%det1=0，不能控

%(c)
ctr_2=ctrb(sys_ss2);
det2=det(ctr_2);
%det2=0，不能控

%(d)
ctr_3=ctrb(sys_ss3);
det3=det(ctr_3);
%det3=0，不能控

%(e)
sys_tf2=tf(sys_ss2);
sys_tf2=minreal(sys_tf2);

%(f)
sys_ss2_new=ss(sys_tf2);
ctr_2_new=ctrb(sys_ss2_new);
det2_new=det(ctr_2_new);
%det2_new=64，所以能控

%(g)
p=[-1+j;-1-j;-10;-10];
[a,b,c,d]=ssdata(sys_ss2_new);
K=acker(a,b,p)
% K=[22 71.56 60 27.02]