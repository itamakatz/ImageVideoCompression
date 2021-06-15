% The easies way to use the parallel toolbox is by using parfor. 
% parffor is an incredibly strong tool as all you need to do to use it is to change the 'for' of a loop to a 'parfor'
% I will show here tow ways to use parfor.

% approach 1 - a simple parfor function
% here we simply use parfor instead of for. 
% the problem is that many times there are dependencies
% additionally, not always can you set the value of an array when using parfor

parallel_1()
parallel_2()

function parallel_1
    results = zeros(2,100);
    parfor i = 1:100
        results(:,i) = [i; factorial(i)];
    end
    
    for i = 1:length(results)
        disp([num2str(results(1,i)) '! = ' num2str(results(2,i))])
    end
end

% approach 2

% the idea is to define what happends after each loop finishes using the 'afterEach' and 'send' keywords 
% and an anomymus function which will be invoked on each call
% note that this approach is limmited by what can be done in the anonymus function and thus is usually used for
% for user interface although it can be used to save things like images

function parallel_2
    q = parallel.pool.DataQueue;
    % define the callback function
    afterEach(q, @(x) (disp([num2str(cell2mat(x(1))) '! = ' num2str(cell2mat(x(2)))])));
    parfor i = 1:100
        data = {i, factorial(i)}
        % whenrver we finish computing, we invoke the afterEach function by calling send
        send(q, data);
    end
end

function val = factorial(n)
    if(n==1), val=n; else, val=n*factorial(n-1); end
end