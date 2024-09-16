%-----------------------------------------------------------------
% phase amplitude estimation using LS
%-----------------------------------------------------------------
%randn('seed',0);
close all
w=randn;
amp=abs(randn);
theta=randn;
NN=100
fs=50000
w=2*pi*(511.456)/50000;

fid= fopen('Rec02_limited.txt','r');
y=fscanf(fid,'%f');
%yy=0.2*sin(w*1.001.*(0:40000-1)+pi/4);
%y=y(100000:end);
sd = 0.1; 
normal_noise = sd* randn(1, length(y));


y=y + transpose(normal_noise);



yy=y(1:fix(length(y)/NN)*NN);
yy=yy-mean(yy);


y=reshape(yy,NN,length(yy)/NN);
fclose(fid);


%------ biuld matrix H -----
H=zeros(NN,2);
VV=0;
amp_v=[];
phase_v=[];
para_x=[]
para_y=[]
for z=1:length(yy)/NN
    for k=1:NN
        H(k,:)=[sin(w*((k-1)+VV)) cos(w*((k-1)+VV))];
    end
    VV=VV+NN;
    para=inv(transpose(H)*H)*transpose(H)*(y(:,z));
    amp1=sqrt(para(1).^2+para(2).^2);
    theta1=atan(para(2)./para(1));
    amp_v=[amp_v amp1];
    phase_v=[phase_v theta1];
    para_y=[para_y para(2)];
    para_x=[para_x para(1)];
end


para_x_g=para_x(1:1000);
para_y_g=para_y(1:1000);

yyy=para_x_g+i.*para_y_g;
H2=transpose(yyy(2:end));
vec2=transpose(yyy(1:end-1));
exp_filter=inv(H2'*H2)*H2'*vec2

hh=[1,-exp_filter'];
yyyy=para_x+i.*para_y;
rr=conv(hh,yyyy);


car=exp_filter.^(0:length(yyyy)-1);
yyyyy=car.*yyyy;
aaaa=yyyyy(10:end)./yyyyy(1:end-9);
ttt=atan(imag(aaaa)./(real(aaaa)));
figure
plot(ttt*180/pi)