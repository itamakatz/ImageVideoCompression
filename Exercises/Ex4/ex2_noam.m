% im = imread('Mona-Lisa.bmp');
global k1 k2 k3 k4 im
im = imread('first_im.bmp');
k1 = 3;
k2 = 2;
k3 = 0;
k4 = 3;

% b=0.314, psnr=32.1175, rate=0.027244

% b=0.326, psnr=31.9315, rate=0.026774
% b=0.332, psnr=31.8325, rate=0.026546

% count = 3
% b_arr = linspace(0.2960,0.35,4)
% b=0.32 % b=0.32, psnr=32.0125, rate=0.027008
% b_arr = [0.2900, 0.2930, 0.2960];
% b_arr=0.05:0.05:0.99;
% figure
% run_benchmark()
% disp('benchmark done')
% plot_rate_psnr(b_arr, false) %Q6
% disp('psnr plot q6 done')
% plot_rate_psnr(b_arr, true)  %Q7
% disp('psnr plot q7 done')
% title('PSNR-Rate')
% xlabel('Rate')
% ylabel('PSNR')
% legend('Benchmark','Our algorithm','Our algorithm with dpcm')

% plot_svd(0.99)

% Ex4

% b = 0.05;
% b = 0.15;
b = 0.32;
% b = 1;

% =================== 2.1 =================== %
% compress_video('short_bw.y4m',b)
% read_images_to_video('short_bw_compressed_new.y4m')

% =================== 2.2 =================== %
compress_video_2_2('short_bw.y4m', 'compressed_2_2_2.y4m',b)

% Explanation:
% -----------------------
% reference_1
% residual = new - reference_1
% compressed_residual = compress(residual)
% reference_2 = reference_1 + compressed_residual
% -----------------------
% A_{i}
% B = New - A_{i}
% C = compress(B)
% A_{i+1} = A_{i} + C
% -----------------------
% Ref_{1} = compress(New_{1})
% Ref_{i+1} = Ref_{i} + compress(New{i+1} - R_{i})
% Ref_{i+1} - Ref_{i} = compress(New{i+1} - R_{i})
% -----------------------

% test_find_ref_block_vec_3_1()

% compress_frames_3_2('short_bw.y4m', 'compressed_3_2.y4m',b)
% compress_frames_4('short_bw.y4m', 'compressed_4.y4m',b, 0.2)

function compress_frames_3_2(path, output_path,b)
    compress_frames_general(path, output_path,b, 0)
end

function compress_frames_4(path, output_path,b, lambda)
    compress_frames_general(path, output_path,b, lambda)
end


function compress_frames_general(path, output_path,b, lambda)
    disp('Running compress_video_3_2...')
    frames = get_frames(path);
    disp('Finished reading frames')
    refs = cell(length(frames),1);
    disp(['Compressing frame 1/' num2str(length(frames))])
%     refs(1) = {cast(compress_frame(cell2mat(frames(1)), b),'double')};
    refs(1) = frames(1);
    imwrite(cast(cell2mat(frames(1)) ,'uint8'), ['Frames_3_2/frame' sprintf('%05d',1) '.bmp'], 'bmp');
    imwrite(cast(cell2mat(refs(1)) ,'uint8'), ['Frames_3_2/refs' sprintf('%05d',1) '.bmp'], 'bmp');
    
    for i=2:length(frames)
        disp(['Compressing frame: ' num2str(i) '/' num2str(length(frames))])
%         compressed_residual = compress_frame(cell2mat(frames(i)) - cell2mat(frames(i-1)), b);
        new_ref_im = get_moved_ref_general(cell2mat(frames(i)), cell2mat(refs(i-1)), lambda);

        compressed_residual = compress_frame(cell2mat(frames(i)) - new_ref_im, b);
        refs(i) = {new_ref_im + compressed_residual};
        
        imwrite(cast(compressed_residual ,'uint8'), ['Frames_3_2/residual_' sprintf('%05d',i) '.bmp'], 'bmp');
        imwrite(cast(cell2mat(frames(i)) ,'uint8'), ['Frames_3_2/frame_' sprintf('%05d',i) '.bmp'], 'bmp');
        imwrite(cast(cell2mat(refs(i)) ,'uint8'), ['Frames_3_2/refs_' sprintf('%05d',i) '.bmp'], 'bmp');
    end

    disp('Compressing finished.')
    
    out_fid = fopen(output_path, 'wb');
    y4m_header = uint8(['YUV4MPEG2 W1920 H1080 F60:1 Ip A0:0 C420jpeg XYSCSS=420JPEG' 10]);
    fwrite(out_fid, y4m_header, 'uint8');

    for i=1:length(refs)
        disp(['Saving frame: ' num2str(i) '/' num2str(length(refs))])
        im = cell2mat(refs(i));
        frame = struct;
        frame.y = cast(im ,'double');
        frame.cr = ones(540,960)*128;
        frame.cb = ones(540,960)*128;
        write_yuv_frame(out_fid, frame);
    end
    fclose(out_fid);
    
    disp(['saved to: ' output_path])
end

function new_ref_im = get_moved_ref_general(ref_im,new_im, lambda)
    
    d = 8;
    [m,n] = size(ref_im);
    rows = m/d;
    cols = n/d;
    
    motion_vectors_struct(rows*cols) = struct('x',-1,'y',-1, 'original_x',-1,'original_y',-1);

    [X,Y] = meshgrid(1:rows, 1:cols);
    X = reshape(X,1,rows*cols);
    Y = reshape(Y,1,rows*cols);
    ALL = [X;Y];

    q = parallel.pool.DataQueue;
    parfor i = 1:length(ALL)
        block_vec = struct;
        current_indices = ALL(:,i)
        block_vec.x=(current_indices(1)-1)*d+1;
        block_vec.y=(current_indices(2)-1)*d+1;
        if(lambda ~= 0)
            motion_vector = find_ref_block_vec_4(ref_im, new_im, block_vec,8,lambda);
        else
            motion_vector = find_ref_block_vec_3_1(ref_im, new_im, block_vec,8);
        end
        motion_vector.original_x = block_vec.x;
        motion_vector.original_y = block_vec.y;
        motion_vectors_struct(i) = motion_vector
    end

    new_ref_im = zeros(size(ref_im));

    for i = 1:length(ALL)
        block_vec = struct;
        block_vec.x=(ALL(1,i)-1)*d+1;
        block_vec.y=(ALL(2,i)-1)*d+1;
        motion_vector = motion_vectors_struct(i);
        new_ref_im(block_vec.x:block_vec.x+7,block_vec.y:block_vec.y+7) = ...
            ref_im(motion_vector.x:motion_vector.x+7,motion_vector.y:motion_vector.y+7);
    end
end

function ref_vector = find_ref_block_vec_4(ref_im, new_im, block_vec, block_size, lambda)

    check_legal = @(check_x,check_y)(check_x + block_size-1 <= size(ref_im,1) && check_y + block_size-1 <= size(ref_im,2) ...
                                    && check_x  >= 1 && check_y >= 1);

    lagrangians = [];
    for x_offset = -block_size:block_size
        for y_offset = -block_size:block_size
            if(check_legal(block_vec.x+x_offset, block_vec.y+y_offset))
                current_mse = immse(ref_im(block_vec.x:block_vec.x+block_size-1,block_vec.y:block_vec.y+block_size-1), ...
                                new_im(block_vec.x+x_offset:block_vec.x+x_offset+block_size-1,block_vec.y+y_offset:block_vec.y+y_offset+block_size-1));
                lx = get_exp_golomb_code(to_unsigned(x_offset),0);
                ly = get_exp_golomb_code(to_unsigned(y_offset),0);
                lagrangian = current_mse + lambda*(length(lx)*length(ly));
                
                lagrangians = [lagrangians ; block_vec.x+x_offset, block_vec.y+y_offset, lagrangian];
            end
        end
    end

    [C,I]  = min(lagrangians(:,3));
    ref_vector = struct;
    ref_vector.x = lagrangians(I(1),1);
    ref_vector.y = lagrangians(I(1),2);
end

% TESTED
function ref_vector = find_ref_block_vec_3_1(ref_im, new_im, block_vec, block_size)

    check_legal = @(check_x,check_y)(check_x + block_size-1 <= size(ref_im,1) && check_y + block_size-1 <= size(ref_im,2) ...
                                    && check_x  >= 1 && check_y >= 1);

    all_mase = [];
    for x_offset = -block_size:block_size
        for y_offset = -block_size:block_size
            if(check_legal(block_vec.x+x_offset, block_vec.y+y_offset))
                all_mase = [all_mase ; block_vec.x+x_offset, block_vec.y+y_offset, ...
                            immse(ref_im(block_vec.x:block_vec.x+block_size-1,block_vec.y:block_vec.y+block_size-1), ...
                            new_im(block_vec.x+x_offset:block_vec.x+x_offset+block_size-1,block_vec.y+y_offset:block_vec.y+y_offset+block_size-1))];
            end
        end
    end

    [C,I]  = min(all_mase(:,3));
    ref_vector = struct;
    ref_vector.x = all_mase(I(1),1);
    ref_vector.y = all_mase(I(1),2);
end

function test_find_ref_block_vec_3_1
    a = [9,9,9,9,9,9;
    9,9,9,9,9,9;
    9,9,1,1,9,9;
    9,9,1,1,9,9;
    9,9,9,9,9,9;
    9,9,9,9,9,9];

    b = [9,9,9,9,9,9;
    9,9,9,9,9,9;
    9,9,9,9,9,9;
    9,9,9,9,9,9;
    9,9,9,9,1,1;
    9,9,9,9,1,1];

    block_vec = struct;
    block_vec.x = 3;
    block_vec.y = 3;

    ref_vector = find_ref_block_vec_3_1(a,b,block_vec,2)
    a(block_vec.x:block_vec.x+1, block_vec.y:block_vec.y+1)
    b(ref_vector.x:ref_vector.x+1, ref_vector.y:ref_vector.y+1)
    % disp(ref_vector.x)
    % disp(ref_vector.y)
end

function compress_video_2_2(path, output_path,b)
    disp('Running compress_video_2_2...')
    frames = get_frames(path);
    disp('Finished reading frames')
    refs = cell(length(frames),1);
    disp(['Compressing frame 1/' num2str(length(frames))])
    refs(1) = {cast(compress_frame(cell2mat(frames(1)), b),'double')};
    imwrite(cast(cell2mat(frames(1)) ,'uint8'), ['Frames_2_2/frame' sprintf('%05d',1) '.bmp'], 'bmp');
    imwrite(cast(cell2mat(refs(1)) ,'uint8'), ['Frames_2_2/refs' sprintf('%05d',1) '.bmp'], 'bmp');

    for i=2:length(frames)
        disp(['Compressing frame: ' num2str(i) '/' num2str(length(frames))])
        % compressed_residual = compress_frame(cell2mat(frames(i)) - cell2mat(frames(i-1)), b);
        compressed_residual = compress_frame(cell2mat(frames(i)) - cell2mat(refs(i-1)), b);
        refs(i) = {cell2mat(refs(i-1)) + compressed_residual};
        
        imwrite(cast(compressed_residual ,'uint8'), ['Frames_2_2/residual_' sprintf('%05d',i) '.bmp'], 'bmp');
        imwrite(cast(cell2mat(frames(i)) ,'uint8'), ['Frames_2_2/frame_' sprintf('%05d',i) '.bmp'], 'bmp');
        imwrite(cast(cell2mat(refs(i)) ,'uint8'), ['Frames_2_2/refs_' sprintf('%05d',i) '.bmp'], 'bmp');
    end

    disp('Compressing finished.')
    
    out_fid = fopen(output_path, 'wb');
    y4m_header = uint8(['YUV4MPEG2 W1920 H1080 F60:1 Ip A0:0 C420jpeg XYSCSS=420JPEG' 10]);
    fwrite(out_fid, y4m_header, 'uint8');

    for i=1:length(refs)
        disp(['Saving frame: ' num2str(i) '/' num2str(length(refs))])
        im = cell2mat(refs(i));
        frame = struct;
        frame.y = cast(im ,'double');
        frame.cr = ones(540,960)*128;
        frame.cb = ones(540,960)*128;
        write_yuv_frame(out_fid, frame);
    end
    fclose(out_fid);
    
    disp(['saved to: ' output_path])
end

function frames = get_frames(path)
    width            = 1920; % [pixels] - Frame Width
    height           = 1080; % [pixels] - Frame Height
    start_frame      = 1; 
    end_frame        = 20;
    bits_per_color   = 8; % 
    chroma_sampling  ='420';
    
    % frames = zeros(1,length(start_frame:end_frame));
    frames = cell(length(start_frame:end_frame),1);
    for frame_idx = start_frame:end_frame
        frame = yuv_read_frame(path, width, height, frame_idx, bits_per_color, chroma_sampling);
        % frames(frame_idx,1) = {cast(frame.y ,'uint8')};
        frames(frame_idx,1) = {frame.y};
    end
end

function compress_video_2_1(path,b)
    frames = get_frames(path)

    q = parallel.pool.DataQueue;
    afterEach(q, @(x) imwrite(cast(cell2mat(x(1)) ,'uint8'), ['Frames/frame_' sprintf('%05d',cell2mat(x(2))) '.bmp'], 'bmp'));
    % parfor i = 1:1
    parfor i = 1:length(frames)
        [compressed_frame, rate, frame_psnr] = compress_frame(cell2mat(frames(i,1)), b);
        data = {compressed_frame, i}
        send(q, data);
    end
end

function read_images_to_video_2_1(output_path)
    Files = dir('./Frames/*');
    fileNames = [];
    for k=1:length(Files)
        if(contains(Files(k).name, '.bmp'))
            fileNames = [fileNames {Files(k).name}];
        end
    end
    fileNames = sort(fileNames);
    
    out_fid = fopen(output_path, 'wb');
    y4m_header = uint8(['YUV4MPEG2 W1920 H1080 F60:1 Ip A0:0 C420jpeg XYSCSS=420JPEG' 10]);
    fwrite(out_fid, y4m_header, 'uint8');

    for k=1:length(fileNames)
        im = imread(['Frames/' cell2mat(fileNames(k))]);
        frame = struct;
        frame.y = cast(im ,'double');
        frame.cr = ones(540,960)*128;
        frame.cb = ones(540,960)*128;
        write_yuv_frame(out_fid, frame);
    end
    fclose(out_fid);
end

function [compressed_frame, rate, frame_psnr] = compress_frame(frame, b)
    k1 = 3;
    k2 = 2;
    k3 = 0;
    k4 = 3;
    num_pix = size(frame,1)*size(frame,2);
    encoded_image=encoder(frame,b,k1,k2,k3,k4,false);
    rate = numel(encoded_image)/(8*num_pix);
    % compressed_frame = cast((decoder(encoded_image,b,k1,k2,k3,k4,false)),'uint8');
    compressed_frame = decoder(encoded_image,b,k1,k2,k3,k4,false);
    frame_psnr = psnr(frame,compressed_frame);
    disp(['b=' num2str(b) ', psnr=' num2str(frame_psnr) ', rate=' num2str(rate)])
end

function run_benchmark()
    global im
    factor = 10:0.95:100;
    psnr_vec = zeros(1,numel(factor));
    rate_vec = zeros(1,numel(factor));
    for i=1:numel(factor)
        [im_psnr,rate]  = benchmark(im,factor(i));
        psnr_vec(i) = im_psnr;
        rate_vec(i) = rate;
    end
    plot(rate_vec,psnr_vec);
end
    
function plot_rate_psnr(b_arr, dpcm)
    global k1 k2 k3 k4 im
    im_psnr = zeros(1,numel(b_arr));
    rate = zeros(1,numel(b_arr));
    num_pix = size(im,1)*size(im,2);
    for i=1:numel(b_arr)
        b = b_arr(i);
        encoded_image=encoder(im,b,k1,k2,k3,k4,dpcm);
        rate(i) = numel(encoded_image)/(8*num_pix);
        % im2=uint8(decoder(encoded_image,b,k1,k2,k3,k4,dpcm));
        im2=decoder(encoded_image,b,k1,k2,k3,k4,dpcm);
        im_psnr(i) = psnr(im,im2);
        disp(['b=' num2str(b) ', psnr=' num2str(im_psnr(i)) ', rate=' num2str(rate(i))])
    end
    hold on
    plot(rate,im_psnr);
end

function plot_svd(b)
    global im
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
end

function encoded_im=encoder(im,b,k1,k2,k3,k4,dpcm)
    % encode an image
    % arg - im: image
    % arg - b: quantisize factor
    % arg - k1: Golomb-Rice order 
    % arg - k2: Golomb-Rice order 
    % arg - k3: Golomb-Rice order 
    % return - encoded_im: encoded image
    % scale image
    % disp(class(im))
    % disp(numel(im))
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
                % dc = de2bi(to_unsigned(evector(evector(1)+2)),8,'left-msb');
                dc = de2bi(to_unsigned(evector(evector(1)+2)),16,'left-msb');
            end
            integers = cell2mat(arrayfun(@(r)get_exp_golomb_code(r,k3),evector(evector(1)+3:end),'UniformOutput',false));
            new_encoded = [num_of_runs runs dc integers];
            encoded_im = [encoded_im new_encoded];
        end
    end
end

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
                integers(1) = to_sign(bi2de(encoded_im(idx:idx+15),'left-msb'));
                % integers(1) = to_sign(bi2de(encoded_im(idx:idx+7),'left-msb'));
                idx = idx+16;
                % idx = idx+8;
            end
            for i=2:num_runs +1
                [integers(i),idx] = decode_exp_golomb_code(encoded_im,k3,idx);
            end
            % recreate zigzag scan
            zscan = get_zigzag_vector(runs,integers);
            % undo zigzag scan and undo quantisize
            block = izigzag_scan(zscan).*b;
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
            % result(i,j).i=i;
            % result(i,j).j=j;
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
    % im = round(abs(double(scale_im).*256 + 128));
    im = double(scale_im).*256 + 128;
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
    % disp(['n size:' num2str(size(n,1)) 'x' num2str(size(n,2))])
    % disp(['m size:' num2str(size(m,1)) 'x' num2str(size(m,2))])
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