clear;

%сеть векторного квантования, обучаемая с учителем, для классификации точек
%в случе, когда классы не линейно разделимы

%отображаю входное множество и эталонное распределение по классам

P(1:2, 1:12)=[0 0.3 -1.3 1.2 -1.2 -0.5 0.7 -1.4 0.3 0.6 0.8 0.5;
              0.7 -1.3 0.8 0.1 0.9 -0.7 -1.5 0.5 0 0.6 -0.7 0.1];  
T(1:12)=[0 0 0 0 0 0 0 0 1 1 0 1];

figure;
plotpv(P,T), grid;

%строю вектор индексов классов, заменив перед этим в векторе 1 на 2, а -1
%на 1.

TInd(1:12)=[1 1 1 1 1 1 1 1 2 2 1 2];

%преобразую индексы в векторы.
TIndc =  full(ind2vec(TInd)) ;

%создаю сеть. Число нейронов конкурентного слоя - 12, скорость обучения -
%0.1.
 
net = newlvq(minmax(P),12,[9/12 3/12]); %создаю квантованную сеть (по умолчанию learnlv1 – LVQ1 обучающая функция)
view(net);

net.trainParam.epochs = 300; %число эпох
net.trainParam.lr = 0.1; %скорость обучения
net = train(net,P,TIndc); %обучение с помощью заданного по умолчанию метода

display(net);
%отображаю структуру сети
display('Метод инициализации сети:');display(net.initFcn);
display('Активационная функция скрытого слоя:');
display(net.layers{1}.transferFcn);
display('Число нейронов в скрытом слое:');display(net.layers{1}.size);
%отображаю параметры обучения
display(net.trainParam.epochs);
display(net.trainParam.show);
display(net.trainParam.showCommandLine);
display(net.trainParam.showWindow);
display(net.trainParam.time);

%проверяю качество обучения с помощью классификации точек области

%область и шаг:
x_min = -1.5; x_max = 1.5; 
y_min = -1.5; y_max = 1.5;
h = 0.1;

nmb_X = ceil(abs(x_max - x_min)/h+1); %количество точек по горизонтали сетки заданной области
nmb_Y = nmb_X; %количество точек по вертикали сетки заданной области

numberPointsAll = nmb_X*nmb_Y; %количество точек в сетке области

xyGrid(1:2, 1 : numberPointsAll) = 0; %вектор входа

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

output = sim(net,xyGrid); %рассчитываю выход сети для всех узлов сетки
Ac = vec2ind(output); %преобразую выходные значения

NewOutput(1:numberPointsAll) = 0; %преобразую индексы классов: 1 в 0, 2 в 1. 
for i = 1:numberPointsAll
    if (Ac(i) == 1 ) NewOutput(i) = 0;
    else NewOutput(i) = 1;
    end
   
end

figure; %вывожу

plotpv(xyGrid,NewOutput);
point = findobj(gca,'type','line');
set(point,'Color','g');
hold on;
plotpv(P,T)
hold off;
