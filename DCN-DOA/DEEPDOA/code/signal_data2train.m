%% ��������
clc
clear variables
close all
M=8;snapshot=256;
f0=1e6;
fc=1e6;
fs=4*f0;
C=M*(M-1);


%%
DOA11=[];
DOA22=[];
k1=[2:2:40];
k=repmat(k1,1,10);
D_start=-60;
D_stop=59;
for l=1:length(k)
    DOA1=D_start:1:D_stop-k(l);   %��һ���źź͵ڶ����źŲ����ķ���
    DOA2=D_start+k(l):1:D_stop;    
    DOA11=[DOA11,DOA1];           %198:[(118+116+����+80��*20/2]*10 = 19800
    DOA22=[DOA22,DOA2];
end
DOA_train=[DOA11;DOA22];

theta=D_start:1:D_stop;
L=length(theta);                  % L=120
A=exp(1i*pi*fc*(0:M-1)'*sind(theta)/f0);
H=zeros(M*M,L);
for i=1:M
    fhi=A*diag(exp(-1i*pi*(i-1)*sind(theta)));
    H((i-1)*M+1:i*M,:)=fhi;
end

S_label=zeros(length(DOA_train),L);
R_est=zeros(length(DOA_train),C);
S_est=zeros(length(DOA_train),L,2);
S_abs=zeros(length(DOA_train),2*L);
for i=1:length(DOA_train)
    [X1,~]=signal_generate(M,snapshot,DOA_train(1,i),f0,fc,fs,1);     %�����ź�
    [X2,~]=signal_generate(M,snapshot,DOA_train(2,i),f0,fc,fs,1);
    temp1=awgn(X1,-10*rand,'measured');            %��Ӹ�˹������
    temp2=awgn(X2,-10*rand,'measured');
    X= temp1+ temp2;
    [R_est(i,:),Rx]=feature_extract_R(X) ;
    temp=H'*vec(Rx);
    temp=temp/norm(temp);
    S_est(i,:,1)=real(temp);          %ʵ��
    S_est(i,:,2)=imag(temp);          %�鲿
    S_abs(i,:)=[real(temp);imag(temp)];%ʵ���鲿����һ��
    S_label(i,round(DOA11(i))+61)=1;  %��[-60��60]�ĽǶȷŵ�[0��120]��ά��
    S_label(i,round(DOA22(i))+61)=1;

end
% % %

close all
plot(theta,S_est(i,:,1))
xlim([-60,60])
hold on
plot(theta,(S_label(i,:)'))
grid on
legend('A','true')


%save('data2_trainlow.mat','R_est','DOA_train',...
%    'S_label','S_est','S_abs')
save('C:\Users\Administrator\Desktop\DEEPDOA\code\data2train\data2_trainlow.mat','R_est','DOA_train',...
    'S_label','S_est','S_abs')
