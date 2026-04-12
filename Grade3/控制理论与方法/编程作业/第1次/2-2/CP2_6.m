%从左到右，从小到大依次对反馈系统进行化简
%(a)
%上方左边的反馈系统
num1=[1 0];
den1=[1 1 2 2];
sys1=tf(num1,den1);
fdnum1=[4 2];
fdden1=[1 2 1];
fdsys1=tf(fdnum1,fdden1);
allsys1=feedback(sys1,fdsys1,-1);

%上方右边的反馈系统
num2=[1];
den2=[1 0 0];
sys2=tf(num2,den2);
fdnum2=[50];
fdden2=[1];
fdsys2=tf(fdnum2,fdden2);
allsys2=feedback(sys2,fdsys2,+1);

%一大块反馈系统
sys3=series(allsys1,allsys2);%上方的两个串联
fdnum3=[1 0 2];
fdden3=[1 0 0 14];
fdsys3=tf(fdnum3,fdden3);
allsys3=feedback(sys3,fdsys3,-1);

%和最前面的串联
num4=[4];
den4=[1];
sys4=tf(num4,den4);

sys=series(sys4,allsys3)

%(b)
pzmap(sys);

%(c)
p=pole(sys);
z=zero(sys);








