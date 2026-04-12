num=[1 -1];
den=[1 5 10];
sys=tf(num,den);
rlocus(sys);
%使系统稳定的p范围是(0,10)