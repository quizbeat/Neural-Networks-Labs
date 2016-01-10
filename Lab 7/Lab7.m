%% PART 1
clear;
clc;

trange = 0 : 0.025 : 2 * pi;
x = ellipse(trange, 0.6, 0.9, 0.2, -0.1, pi / 8);
xseq = con2seq(x);

plot(x(1, :), x(2, :), '-r', 'LineWidth', 2);

net = feedforwardnet(1, 'trainlm');
net.layers{1}.transferFcn = 'purelin';

net = configure(net, xseq, xseq);
net = init(net);
view(net);

net.trainParam.epochs = 10000;
net.trainParam.goal = 10e-5;
net = train(net, xseq, xseq);

yseq = sim(net, xseq);
y = cell2mat(yseq);

plot(x(1, :), x(2, :), '-r', y(1, :), y(2, :), '-b', 'LineWidth', 2);


%% PART 2
r = 5;
phi = 0.01 : 0.025 : 11 * pi / 6;
x = [r * cos(phi); r * sin(phi)];
xseq = con2seq(x);

plot(x(1, :), x(2, :), '-r', 'LineWidth', 2);

net = feedforwardnet([10 1 10], 'trainlm');

net = configure(net, xseq, xseq);
net = init(net);
view(net);
net.trainParam.epochs = 2000;
net.trainParam.goal = 10e-5;
net = train(net, xseq, xseq);

yseq = sim(net, xseq);
y = cell2mat(yseq);

plot(x(1, :), x(2, :), '-r', y(1, :), y(2, :), '-b', 'LineWidth', 2);


%% PART 3
r = 5;
phi = 0.01 : 0.025 : 11 * pi / 6;
x = [r * cos(phi); r * sin(phi); phi];
xseq = con2seq(x);

plot3(x(1, :), x(2, :), x(3, :), '-r', 'LineWidth', 2);

net = feedforwardnet([10 2 10], 'trainlm');

net = configure(net, xseq, xseq);
net = init(net);
view(net);
net.trainParam.epochs = 1000;
net.trainParam.goal = 10e-5;
net = train(net, xseq, xseq);

yseq = sim(net, xseq);
y = cell2mat(yseq);

plot3(x(1, :), x(2, :), x(3, :), '-r', y(1, :), y(2, :), y(3, :), '-b', 'LineWidth', 2);