fs=150;N=100;%采样频率、采样点数
ds=1/fs;
t=(0:N-1)*ds;
A=1;P=2*pi;
y=A*square(P*(t+0.05),10);
Y=fft(y);
fy=(abs(Y)*2)/N;
subplot(1,1,1);
stem(fy);
axis([0 30 0 2]);
xlabel('频率');
ylabel('幅值');