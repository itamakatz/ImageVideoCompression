%% test code
%load image
im = imread('Mona-Lisa.bmp');
k1 = 3;
k2 = 2;
k3 = 0;
k4 = 3;
% b = 0.01:0.1:0.99;
% dpcm = true;
% for i=1:10
%     encoded_image=encoder(im,b(i),k1,k2,k3,k4,dpcm);
%     im2=decoder(encoded_image,b(i),k1,k2,k3,k4,dpcm);
%     imwrite(uint8(im2),'b_'+string(b(i))+'.bmp')
% end
% plot benchmark psnt-bitrate
factor = 10:0.95:100;
psnr_vec = zeros(1,numel(factor));
rate_vec = zeros(1,numel(factor));
for i=1:numel(factor)
    [im_psnr,rate]  = benchmark(im,factor(i));
    psnr_vec(i) = im_psnr;
    rate_vec(i) = rate;
end
figure(1)
plot(rate_vec,psnr_vec);

% plot rate-psnr
dpcm = false;
b_arr=0.05:0.01:0.99;
im_psnr = zeros(1,numel(b_arr));
rate = zeros(1,numel(b_arr));
num_pix = size(im,1)*size(im,2);
for i=1:numel(b_arr)
    b = b_arr(i);
    encoded_image=encoder(im,b,k1,k2,k3,k4,dpcm);
    rate(i) = numel(encoded_image)/(8*num_pix);
    im2=uint8(decoder(encoded_image,b,k1,k2,k3,k4,dpcm));
    im_psnr(i) = psnr(im,im2);
end
hold on
plot(rate,im_psnr);

dpcm = true;
im_psnr = zeros(1,numel(b_arr));
rate = zeros(1,numel(b_arr));
num_pix = size(im,1)*size(im,2);
for i=1:numel(b_arr)
    b = b_arr(i);
    encoded_image=encoder(im,b,k1,k2,k3,k4,dpcm);
    rate(i) = numel(encoded_image)/(8*num_pix);
    im2=uint8(decoder(encoded_image,b,k1,k2,k3,k4,dpcm));
    im_psnr(i) = psnr(im,im2);
end
hold on
plot(rate,im_psnr);
title('PSNR-Rate')
xlabel('Rate')
ylabel('PSNR')
legend('Benchmark','Our algorithm','Our algorithm with dpcm')

% plot svd of dct 
% scale image
scaled_im = scale_im(im);
% get 8X8 block from the image 
struct = bin_mat(scaled_im,8); 
[row,col] = size(struct);
zigzag_stream = zeros(row*col,64);
i = 0;
for r=1:row
    for c=1:col
        i = i +1 ;
        % apply sct on current block
        coff_block = dct2(struct(r,c).submat);
        % quantisize block
        coff_block = round(coff_block./b);
        % apply zigzag scan on block
        zigzag_stream(i,:) = zigzag_scan(coff_block);
    end
end
std_plot = zeros(1,64);
for k=1:64
    std_plot(k) = std2(zigzag_stream(:,1:k));
end
figure(2)
plot(1:64,std_plot);
title('STD of DCT coefficients-Zigzag Index')
xlabel('Index')
ylabel('STD')
%%
function encoded_im=encoder(im,b,k1,k2,k3,k4,dpcm)
    % encode an image
    % arg - im: image
    % arg - b: quantisize factor
    % arg - k1: Golomb-Rice order 
    % arg - k2: Golomb-Rice order 
    % arg - k3: Golomb-Rice order 
    % return - encoded_im: encoded image
    % scale image
    scaled_im = scale_im(im);
    % get 8X8 block from the image 
    struct = bin_mat(scaled_im,8);
    % apply dct on each block and get it's coeff_vec
    [row,col] = size(struct);
    encoded_im = [get_golomb_rice_code(row,k1) get_golomb_rice_code(col,k1)]; 
    curr_dc = 0;
    for r=1:row
        for c=1:col
            % apply sct on current block
            coff_block = dct2(struct(r,c).submat);
            % quantisize block
            coff_block = round(coff_block./b);
            % apply zigzag scan on block
            zigzag_stream = zigzag_scan(coff_block);
            % get equivalent vector
            evector = get_equivalent_vector(zigzag_stream);
            % apply entropy coding 
            num_of_runs = get_golomb_rice_code(evector(1),k1);
            runs = cell2mat(arrayfun(@(r)get_golomb_rice_code(r,k2),evector(2:evector(1)+1),'UniformOutput',false));
            % check if need to apply dpcm
            if(dpcm)
                dc = get_exp_golomb_code(evector(evector(1)+2)-curr_dc,k4);
                curr_dc = evector(evector(1)+2);
            else
                dc = de2bi(to_unsigned(evector(evector(1)+2)),8,'left-msb');
            end
            integers = cell2mat(arrayfun(@(r)get_exp_golomb_code(r,k3),evector(evector(1)+3:end),'UniformOutput',false));
            new_encoded = [num_of_runs runs dc integers];
            encoded_im = [encoded_im new_encoded];
        end
    end
end
%%
% function new_val = denormalize(val)
% 	assert(val >= -128 && val <= 127)
% 	% assert(val >= -0.5 && val < 0.5)
% 	new_val = val + 128;
% 	% new_val = (val + 0.5)*256;
%     % new_val = round(abs(double(val).*256 + 128));
%     new_val = cast(new_val ,'uint8');
% end
function im=decoder(encoded_im,b,k1,k2,k3,k4,dpcm)
    % decode encoded image
    % arg - encoded_im: encoded image
    % arg - b: quantisize factor
    % arg - k1: Golomb-Rice order 
    % arg - k2: Golomb-Rice order 
    % arg - k3: exp Golomb order 
    % return - im: decoded image
    
    % read struct size
    [row,idx] = decode_golomb_rice_code(encoded_im,k1,1);
    [col,idx] = decode_golomb_rice_code(encoded_im,k1,idx);
    im_struct = struct;
    curr_dc = 0;
    % decode each block
    for r=1:row
        for c=1:col
            % read number of runs
            [num_runs,idx] = decode_golomb_rice_code(encoded_im,k1,idx);
            % read runs
            runs = zeros(1,num_runs);
            for i=1:num_runs
                [runs(i),idx] = decode_golomb_rice_code(encoded_im,k2,idx);
            end
            % read integers
            integers = zeros(1,num_runs +1);
            % read dc
            if(dpcm)
                [integers(1),idx] = decode_exp_golomb_code(encoded_im,k4,idx);
                integers(1) = integers(1) + curr_dc;
                curr_dc = integers(1);
            else
                integers(1) = to_sign(bi2de(encoded_im(idx:idx+7),'left-msb'));
                idx = idx+8;
            end
            for i=2:num_runs +1
                [integers(i),idx] = decode_exp_golomb_code(encoded_im,k3,idx);
            end
            % recreate zigzag scan
            zscan = get_zigzag_vector(runs,integers);
            % undo zigzag scan and undo quantisize
            izigzag = izigzag_scan(zscan)
            % block = arrayfun(@(x) denormalize(x), mat);
            block = izigzag.*b;
            % block = izigzag_scan(zscan).*b;
            % apply idct
            block = idct2(block);
            im_struct(r,c).submat = block;
        end
    end
    % unbin image
    im = unbin_mat(im_struct);
    % rescale image
    im = rescale_im(im);
end
%%
function evector=get_equivalent_vector(zigzag_stream)
    % create equivalent vector of the zigzag scan
    % arg - zigzag_stream: 
    % return - evector: equivalent vector
    % find las non zero integer
    last_non_zero = find(zigzag_stream,1,'last');
    if isempty(last_non_zero)
        last_non_zero = 1;
    end
    zigzag_stream = zigzag_stream(1:last_non_zero);
    % calc number of runs
    % conver to string
    s = sprintf('%d',zigzag_stream(2:end) ~=0);
    if(isempty(s))
        evector = [0 zigzag_stream];
    else    
        % split to 1's and 0's
        txt_scan=textscan(s,'%s','delimiter','1');
        txt_scan = txt_scan(:);
        % calc length of each zeros run
        runs = cellfun(@(t)length(t), txt_scan{1});
        % calc integers
        integers = zigzag_stream(zigzag_stream ~= 0);
        % in case of zero as integer
        if(zigzag_stream(1) == 0)
            integers = [0 integers];
        end
        evector = [length(runs) runs' integers];
    end
end
%%
function zvector=get_zigzag_vector(runs,integers)
    % create zigzag scan vector from runs of zeroes and integers
    % arg - runs: 
    % arg - integers: 
    % return - zvector: 
    zvector = zeros(1,64);
    zvector(1) = integers(1);
    curr_idx = 2;
    for i=1:numel(runs)
        run = runs(i);
        curr_idx = curr_idx + run;
        zvector(curr_idx) = integers(i+1);
        curr_idx = curr_idx +1;
    end
end
%%
function [im_psnr,rate] = benchmark(im,factor)
    % create benchmark of jpeg
    % arg - im: image
    % arg - factor: quality factor
    % return - psnr,rate
    % compress the image using matlab's jpeg
    imwrite(im,'jpg_mona_lisa.jpg','jpg','quality',factor);
    % calc psnr
    [im_psnr,~] = psnr(im,imread('jpg_mona_lisa.jpg'));
    % calc rate
    info = imfinfo('jpg_mona_lisa.jpg');
    rate = info.FileSize/(8*info.Width*info.Height);
end
%%
function result=bin_mat(mat,d)
    % divide matrix to sub matrices 
    % arg - mat:
    % return - result: struct of sub matrices
    [m,n] = size(mat);
    % split the matrix into cell of size (dXd)
    rows = ones(1,m/d)*d;
    cols = ones(1,n/d)*d;
    cells = mat2cell(mat,rows,cols);
    [a,b] = size(cells);
    result(a,b).submat = zeros(d);
    % create struct of matrices
    for i=1:a
        for j=1:b
            result(i,j).submat = cells{i,j};
        end
    end
end
%%
function result=unbin_mat(struct)
    % combine matrix to sub matrices to matrix 
    % arg - struct:
    % return - result: matrix

    % get dimansions
    [row,col] = size(struct);
    [d,~] = size(struct(1,1).submat);

    % create new matrix
    result = zeros(row*d,col*d,'double');
    curr_row = 1;
    % unbin the matrix
    for i=1:row
        curr_col = 1;
        for j=1:col
            curr_mat = struct(i,j).submat;
            result(curr_row:curr_row + d -1,curr_col:curr_col + d -1) = curr_mat;
            curr_col = curr_col + d;
        end
        curr_row = curr_row + d;
    end
end
%%
function scale_im=scale_im(im)
    % scale image betweem [-0.5,0.5)
    % arg - im: image
    % return - scale_im: rescale image
    scale_im = (double(im) -128)./256;
end
%%
function im=rescale_im(scale_im)
    % rescale image
    % arg - scale_im: scaled image
    % return - im: image
    im = round(abs(double(scale_im).*256 + 128));
end
%% 
function scan_res=zigzag_scan(im)
    % apply zigzag scan on an image
    % arg - im: image
    % return - scan_res: vector of the scan
    zigzag_idx = [1 2 9 17 10 3 4 11 18 25 33 26 19 12 5 6 13 20 ...
        27 34 41 49 42 35 28 21 14 7 8 15 22 29 36 43 50 57 58 51 ...
        44 37 30 23 16 24 31 38 45 52 59 60 53 46 39 32 40 47 54 61 62 55 48 56 63 64];
    % matlab works on columns so we take the traspose of the image
    im = im';
    scan_res = im(zigzag_idx);
end
%% 
function res=get_exp_golomb_code(n,k)
            % calc golomb code of a number
            % arg - n:
            % arg - k: order
            % return - res: 
            
            n = to_unsigned(n);
            % calc q and r 
            q = floor(log2(double(n+2^k))) -k;
            q_res = [zeros(1,q),1];
            q_res = logical(q_res);
            r_res = de2bi(n+2^k,'left-msb');
            r_res = logical(r_res(2:end));
            res = [q_res,r_res];
end
%%
function res = to_unsigned(n)
    % convert sign number to unsign
    % arg - n:
    % return - res: 
    if(n > 0)
        res = 2*n-1;
    else
        res = -2*n;
    end
end
%%
function n=to_sign(n)
    if(mod(n,2) == 0)
        n = n/-2;
    else
        n = (n+1)/2;
    end
end
%%
function [n,idx]=decode_exp_golomb_code(stream,k,idx)
    % decode a number
    % arg - stream:
    % arg - k:
    % arg - idx:
    % return - n,idx:
    
    % count zeroes
    num_zero = find(stream(idx:end),1,'first') -1;
    idx = idx + num_zero;
    n = stream(idx:idx + num_zero+k);
    % convert back decimal
    n = cast(bi2de(n,'left-msb'),'double') - 2^k;
    % conver to sign
    n = to_sign(n);
    idx = idx + num_zero +k+1;
end
%%
function res=get_golomb_rice_code(n,m)
    % calc golomb rice code of a number
    % arg - n:
    % arg - m: 
    % return - res: 

    % calc q and r 
    q = floor(n/m);
    r = mod(n,m);
    q_res = [zeros(1,q),1];
    q_res = logical(q_res);
    b = floor(log2(m));
    if(r < (2^(b+1)-m))
        r_res = de2bi(r,b,'left-msb');
    else
        r = r -m + 2^(b+1);
        r_res = de2bi(r,b+1,'left-msb');
    end
    r_res = logical(r_res);
    res = [q_res,r_res];
end
%%
function [res,idx]=decode_golomb_rice_code(stream,m,idx)
    % calc golomb rice code of a number
    % arg - stream: stream of bits
    % arg - idx: start index 
    % return - res,idx: 
    
    q = find(stream(idx:end),1,'first') -1;
    idx = idx+q;
    b = floor(log2(m));
    r = bi2de(stream(idx+1:idx + b),'left-msb');
    if(r < 2^(b+1) -m)
        idx = idx + b +1;
    else
        r = bi2de(stream(idx+1:idx + b+1),'left-msb') -2^(b+1) + m;
        idx = idx + b + 2;
    end
    res = q * m + r;
end
%%
function original_im=izigzag_scan(vec)
    % apply reverse zigzag scan on an image
    % arg - im: image
    % return - scan_res: vector of the scan
    izigzag_idx = [1 2 6 7 15 16 28 29 3 5 8 14 17 27 30 ...
        43 4 9 13 18 26 31 42 44 10 12 19 25 32 41 45 54 11 20 24 33 40 46 53 55 21 23 34 39 47 52 ...
        56 61 22 35 38 48 51 57 60 62 36 37 49 50 58 59 63 64];
    original_im = vec(izigzag_idx);
    original_im = reshape(original_im,8,8)';
end