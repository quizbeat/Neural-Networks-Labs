clear;

%���� ���������� �����������, ��������� � ��������, ��� ������������� �����
%� �����, ����� ������ �� ������� ���������

%��������� ������� ��������� � ��������� ������������� �� �������

P(1:2, 1:12)=[0 0.3 -1.3 1.2 -1.2 -0.5 0.7 -1.4 0.3 0.6 0.8 0.5;
              0.7 -1.3 0.8 0.1 0.9 -0.7 -1.5 0.5 0 0.6 -0.7 0.1];  
T(1:12)=[0 0 0 0 0 0 0 0 1 1 0 1];

figure;
plotpv(P,T), grid;

%����� ������ �������� �������, ������� ����� ���� � ������� 1 �� 2, � -1
%�� 1.

TInd(1:12)=[1 1 1 1 1 1 1 1 2 2 1 2];

%���������� ������� � �������.
TIndc =  full(ind2vec(TInd)) ;

%������ ����. ����� �������� ������������� ���� - 12, �������� �������� -
%0.1.
 
net = newlvq(minmax(P),12,[9/12 3/12]); %������ ������������ ���� (�� ��������� learnlv1 � LVQ1 ��������� �������)
view(net);

net.trainParam.epochs = 300; %����� ����
net.trainParam.lr = 0.1; %�������� ��������
net = train(net,P,TIndc); %�������� � ������� ��������� �� ��������� ������

display(net);
%��������� ��������� ����
display('����� ������������� ����:');display(net.initFcn);
display('������������� ������� �������� ����:');
display(net.layers{1}.transferFcn);
display('����� �������� � ������� ����:');display(net.layers{1}.size);
%��������� ��������� ��������
display(net.trainParam.epochs);
display(net.trainParam.show);
display(net.trainParam.showCommandLine);
display(net.trainParam.showWindow);
display(net.trainParam.time);

%�������� �������� �������� � ������� ������������� ����� �������

%������� � ���:
x_min = -1.5; x_max = 1.5; 
y_min = -1.5; y_max = 1.5;
h = 0.1;

nmb_X = ceil(abs(x_max - x_min)/h+1); %���������� ����� �� ����������� ����� �������� �������
nmb_Y = nmb_X; %���������� ����� �� ��������� ����� �������� �������

numberPointsAll = nmb_X*nmb_Y; %���������� ����� � ����� �������

xyGrid(1:2, 1 : numberPointsAll) = 0; %������ �����

x = x_min;
y = y_min;
crntIndx = 1;
for j = 1 : nmb_Y
    for i = 1 : nmb_X
       xyGrid(1, crntIndx) = x + (i-1)*h;
       xyGrid(2, crntIndx) = y + (j-1)*h;
       crntIndx =  crntIndx + 1;
    end
end

output = sim(net,xyGrid); %����������� ����� ���� ��� ���� ����� �����
Ac = vec2ind(output); %���������� �������� ��������

NewOutput(1:numberPointsAll) = 0; %���������� ������� �������: 1 � 0, 2 � 1. 
for i = 1:numberPointsAll
    if (Ac(i) == 1 ) NewOutput(i) = 0;
    else NewOutput(i) = 1;
    end
   
end

figure; %������

plotpv(xyGrid,NewOutput);
point = findobj(gca,'type','line');
set(point,'Color','g');
hold on;
plotpv(P,T)
hold off;
