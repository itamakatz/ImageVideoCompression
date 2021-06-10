q = parallel.pool.DataQueue;
afterEach(q, @disp);
parfor i = 1:3
    send(q, i); 
end;