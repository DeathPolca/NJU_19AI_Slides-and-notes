%(a)
num=[1];
den=[1 5 6];
sys1=tf(num,den);
rlocus(sys1);
hold on
plot([-0.4 -0.4],[-6 6],'--',...
[0 -6*tan(36.2*pi/180)],[0 6],'--',...    
[0 -6*tan(36.2*pi/180)],[0 -6],'--')
hold off
[K1,p1]=rlocfind(sys1);
%K1=11.392,p1=-2.503+3.338j;

%(b)
num=[1];
den=[1 5 6 0];
sys2=tf(num,den);
rlocus(sys2);
hold on
plot([-0.4 -0.4],[-6 6],'--',...
[0 -6*tan(36.2*pi/180)],[0 6],'--',...    
[0 -6*tan(36.2*pi/180)],[0 -6],'--')
hold off
[K2,p]=rlocfind(sys2)
% K2=4.093,p=-0.669+0.821j

%(c)
num=[1 1];
den=[1 5 6 0];
sys3=tf(num,den);
rlocus(sys3);
hold on
plot([-0.4 -0.4],[-6 6],'--',...
[0 -6*tan(36.2*pi/180)],[0 6],'--',...    
[0 -6*tan(36.2*pi/180)],[0 -6],'--')
hold off
[K3,p]=rlocfind(sys3)
% K3=9.2516,p=-2.0695+2.7387j

%(d)
K1=11.392;
K2=4.093;
K3=9.2516;
t=[0:0.01:10];
sys1_all=feedback(sys1*K1,[1]);
sys2_all=feedback(sys2*K2,[1]);
sys3_all=feedback(sys3*K3,[1]);
[y1,t]=step(sys1_all,t);
[y2,t]=step(sys2_all,t);
[y3,t]=step(sys3_all,t);
plot(t,y1,':',t,y2,'--',t,y3,'g'),grid
xlabel('t/s'),ylabel('y(t)')
legend('y1','y2','y3')

%(e)比例控制器有明显稳态误差，积分控制器和比例积分控制器
%没有稳态误差，但积分控制器仅在K<30时稳定，比例积分对所有
%K>0均稳定，并且响应也更快