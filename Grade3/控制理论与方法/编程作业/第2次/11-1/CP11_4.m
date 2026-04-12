%(a)
A=[0 1 0 0 0;-0.1 -0.5 0 0 0;0.5 0 0 0 0;0 0 10 0 0;0.5 1 0 0 0];
B=[0;1;0;0;0];
C=[0 0 0 1 0];
D=[0];
sys_ss=ss(A,B,C,D);
Pc=ctrb(sys_ss);
det_Pc=det(Pc)
% det_Pc=0,所以不能控

%(b)
sys_tf=tf(sys_ss);
sys_tf1=minreal(sys_tf);
sys_ss1=ss(sys_tf1);

%(c)
Pc1=ctrb(sys_ss1);
det_Pc1=det(Pc1);
%det_Pc1=16,所以能控

%(d)
eva=eig(sys_ss1)
%不稳定，因为有一对极点在虚轴上

%(e)
%复杂系统（状态多）不可控，但经过零极点对消后得到的
%简单一点的系统（状态少）可以变成可控。