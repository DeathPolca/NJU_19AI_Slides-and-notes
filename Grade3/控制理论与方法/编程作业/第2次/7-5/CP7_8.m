% 1+Kd(Kp/Kd+s)/Js^2=0
num=[1 5];
den=[1 0 0];
sys=tf(num,den);
rlocus(sys);
% Kd/J=7.1075, Kp/J=5Kd/J=35.5375