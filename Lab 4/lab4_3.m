clear;

%карта Кохонена для нахождения одного из решений задачи коммивояжера

%задаю 20 случайных двухмерных входных векторов
T(1:2, 1:20)= 0;
for i = 1 : 2
    for j = 1 : 20
        T(i,j)=3.0*rand-1.5; 
    end
end

figure;
hold on;
plot(T(1,:),T(2,:),'-v','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize', 7), grid;
title('Точки 20 городов');
xlabel('Ось X');
ylabel('Ось Y'); 
hold off;

%создаю сеть Кохонена, нейроны которой образуют одномерную цепочку

net = newsom(T, 20);

view(net);

net.trainParam.epochs = 600; %число эпох
net = train(net,T); %обучение заданным по умолчанию методом

%отображаю структуру сети и проведенное обучение
display('Метод инициализации сети:');display(net.initFcn);
display('Активационная функция скрытого слоя:');
display(net.layers{1}.transferFcn);
display('Число нейронов в скрытом слое:');display(net.layers{1}.size);
%параметры обучения:
display(net.trainParam.epochs);
display(net.trainParam.show);
display(net.trainParam.showCommandLine);
display(net.trainParam.time);

%отображаю координаты городов и центры кластеров, сгенерированные сетью

figure;
hold on;
plot(T(1,:),T(2,:), 'V','MarkerEdgeColor','k', 'MarkerFaceColor','g', 'MarkerSize',14), grid;
plotsom(net.iw{1,1},net.layers{1}.distances);
title('Точки 20 городов и кластеры сети');
legend('Города', 'Связи между кластерами', 'Центры кластеров', -1);
hold off;

