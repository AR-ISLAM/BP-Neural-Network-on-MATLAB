%% Developer Section
% Project Title: Design and Implementation of BP Neural Network
% Developer: MD ARRAHMANUL ISLAM (伊兰沐)
% Student Id: 231610300005
% School of Automation, Jiangsu University of Science and Technology
%
% Contact Info: ar.islam.web@gmail.com, https://www.linkedin.com/in/ar-islam/
% Github: https://github.com/AR-ISLAM

% Inspired by the book "Intelligent Control Design and MATLAB Simulation"
% by Jinkun Liu

%% Clearing Datas on Matlab
clc;
clear;
close all;

%% Initialization
epoch = 1000;               % Number of BP loops in the neural network
neurons = 6;                % Number of neurons in the system
eta = 0.50;                 % BP Learning Rate
alpha = 0.05;               % Momentum Factor

wij = rands(2, 6);          % The random weights from input layer to hidden layer
wij_1 = wij;                % Initializing the previous weight value
wjo = rands(1,6);           % The random weights from hidden layer to output layer
wjo_1 = wjo;                % Initializing the previous weight value

dwij = 0 * wij;             % Initializing the delta wij values
dwjo = 0 * wjo;             % Initializing the delta wjo values

u_1 = 0;                    % Previous value of the input
y_1 = 0;                    % Previous value of the output
ts = 0.001;                 % Simulation sample time

time = size(epoch);         % Preallocating the time frame
u = zeros(1,epoch);         % Preallocating the size of input
y = zeros(1,epoch);         % Preallocating the suze of output
N_out = zeros(1,neurons);   % Preallocating the size of output of the nurons
yo = zeros(1, epoch);       % Preallocating the size of actual output
e = zeros(1, epoch);        % Preallocating the size of errors

for k = 1:epoch

    %% Calculating the Inputs & Outputs

    time(k) = k*ts;                         % Current simulation time
    u(k) = 0.5 * sin(6 * pi * time(k));     % Current input value for current time
    y(k) = (u_1 - (0.9*y_1)) / (1 + y_1^2); % Current output value for current input
    
    
    %% Neural Network Operation

    x = [u(k) y(k)];                        % input vector x = [u(k) y(k)]
    
    % Feed Forward Process
    for j = 1:neurons
        N_ws = x * wij(:,j);                % Calculating the weighted sum at each neuron
        N_out(j) = Sigmoid(N_ws);           % Calculating the output at each neuron
    end

    yo(k) = N_out * wjo';                   % Calculating the actual output

    % Calculating error
    e(k) = y(k) - yo(k); 

    %% Learning Algorith of BP Neural Network
    
    % Calculating the delta wjo and delta dwij
    dwjo = eta * e(k) * N_out;
    for i = 1:2
        for j = 1:neurons
            dwij(i,j) = eta * e(k) * wjo(j) * N_out(j) * (1 - N_out(j)) * x (i); 
        end 
    end 

    % Updating the weights
    wjo = wjo + dwjo + alpha * (wjo - wjo_1);
    wij = wij + dwij + alpha * (wij - wij_1);

    wjo_1 = wjo;                            % Setting the previous wjo for the next loop
    wij_1 = wij;                            % Setting the previous wij for the next loop
    u_1 = u(k);                             % Setting the previous input value for the next loop
    y_1 = y(k);                             % Setting the previous output value for the next loop
    
end

% Plotting
figure(1)
plot(time, y, 'r', time, yo, 'b');
xlabel('Time');
ylabel('Desired Y and Actual Y');

figure(2);
plot(time, e , 'r');
xlabel('Time');
ylabel('Error');


%% All Functions
function y = Sigmoid(x)
    y = 1/(1+exp(-x));
end