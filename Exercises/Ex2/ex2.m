% a = [ 25 0 0 -13 11 0 7 0 0 0 0 ]
% a = [ 5 0 1 0 0 3 0 0 0 ]
% k1 = 2
% k2 = 2
% k3 = 2
% N = 3
% encoded = encode_equivalent_vector(a, k1,k2,k3)
% vec = decode_equivalent_vector(encoded, N,k1,k2,k3)
% assert(isequal(vec,a))
% benchmark()
% golomb_rice_test()
% exp_golomb_test()
% normalize_test()
% Q4()
Q6()
% Q7()
% equivalent_vector_test()
% golomb_rice(8,2)
% golomb_rice(1,2)

% test_dct()

% count = 20
% b = 1;
% for i = 1:count
% 	current_b = b*2^(i-1);
% 	% disp(num2str(current_b,'%.4g'))
%     disp(num2str(current_b))
%     % disp(num2str(current_b,'%.4E'))
% end

% =========== B: ZigZag =========== %

% Tested
function zigZagArray = zigzag(mat)
	ind = reshape(1:numel(mat), size(mat));   %# indices of elements
	ind = fliplr(spdiags(fliplr(ind)));       %# get the anti-diagonals
	ind(:,1:2:end) = flipud(ind(:,1:2:end));  %# reverse order of odd columns
	ind(ind==0) = [];                         %# keep non-zero indices
	zigZagArray = mat(ind);
end

% Tested
function mat = izigzag(zigZagArray, imSize)
	M = 1:length(zigZagArray);
	M = reshape(M, imSize);
	indices = zigzag(M);
	mat = zeros(imSize);
	for i = 1:length(indices)
		index = indices(i);
		mat(fix((index-1)/imSize(2))+1,rem(index-1,imSize(2))+1) = zigZagArray(i);
	end
	mat = reshape(mat.', imSize);
end

% Tested
function success = zigzag_test()
	M = randi(100, [5 6])
	zigZagArray = zigzag(M);
	restoreM = izigzag(zigZagArray, size(M))
	
	success = isequal(M, restoreM)
	assert(success)
end

% =========== E: ZigZag =========== %
% =========== B: Blocking =========== %

% Tested
function structMat = splitMat2Struct(mat, blockSizeArray)
	if(size(blockSizeArray) == [1,1])
		blockSizeArray = [blockSizeArray, blockSizeArray];
	end 

	matSize = size(mat);

	reshapedMat = permute(reshape(mat, blockSizeArray(1), matSize(1) / blockSizeArray(1), blockSizeArray(2), matSize(2) / blockSizeArray(2)), [2, 4, 1, 3]);
	structMat(matSize(1)/blockSizeArray(1),matSize(2)/blockSizeArray(2)).submat = zeros(size(reshapedMat(1,1,:,:)), class(size(reshapedMat(1,1,:,:))));
	structMat(matSize(1)/blockSizeArray(1),matSize(2)/blockSizeArray(2)).origin = [0,0];
	for i = 1:matSize(1)/blockSizeArray(1)
		for j = 1:matSize(2)/blockSizeArray(2)
			bla = reshapedMat(i,j,:,:);
			structMat(i,j).submat = reshape(bla, blockSizeArray(1), blockSizeArray(2));
			structMat(i,j).origin = [1 + (i-1)*blockSizeArray(1), 1 + (j-1)*blockSizeArray(2)];
		end
	end
end

% Tested
function mat = structMat2Mat(structMat)
	structMatSize = size(structMat);
	blockSize = [size(structMat(1,1).submat,1), size(structMat(1,1).submat,2)];
	mat = zeros([structMatSize(1)*blockSize(1), structMatSize(2)*blockSize(2)], class(structMat(1,1).submat));
	for i = 1:structMatSize(1)
		for j = 1:structMatSize(2)
			mat(structMat(i,j).origin(1):structMat(i,j).origin(1)+blockSize(1)-1,...
				structMat(i,j).origin(2):structMat(i,j).origin(2)+blockSize(2)-1) = structMat(i,j).submat;
		end
	end
end

% =========== E: Blocking =========== %
% =========== B: Normalization =========== %

% Tested
function new_val = normalize(val)
	assert(val >= 0 && val <= 255)
    % val = cast(val,'int8');
	% new_val = val - 128;
	% new_val = val/256 - 0.5;
	new_val = (double(val) -128)./256;
end

% Tested
function new_val = denormalize(val)
	assert(val >= -128 && val <= 127)
	% assert(val >= -0.5 && val < 0.5)
	% new_val = val + 128;
	% new_val = (val + 0.5)*256;
    new_val = round(abs(double(val).*256 + 128));
    new_val = cast(new_val ,'uint8');
end

% Tested
function success = normalize_test()
	val = randi([0,255]);
    val = cast(val,'uint8');
	restore_val = denormalize(normalize(val));
	success = isequal(restore_val, val);
	assert(success)

	val = 0;
	restore_val = denormalize(normalize(val));
	success = isequal(restore_val, val);
	assert(success)

	val = 255;
	restore_val = denormalize(normalize(val));
	success = isequal(restore_val, val);
	assert(success)
end

% =========== E: Normalization =========== %
% =========== B: MSE / PSNR / Rate =========== %

% Tested
function error_val = MSE(im1,im2)
	error_val = immse(im1,im2);
end
% Tested
function val = PSNR(im1,ref)
	val = psnr(im1,ref);
end
% Tested
function rate = image_rate(image_path)
	imInfo = imfinfo(image_path);
	rate = imInfo.FileSize / (imInfo.BitDepth * imInfo.Width * imInfo.Height);
end

% =========== S: MSE / PSNR / Rate =========== %

% =========== B: Benchmark =========== %

% Tested
function benchmark()
	im = imread('Mona-Lisa.bmp');
	psnr_array = zeros([1,101]);
	rate_array = zeros([1,101]);
	for i = 0:100
		imwrite(im,'Mona-Lisa.jpg','jpg','quality',i);
		imCompressed = imread('Mona-Lisa.jpg');
		psnr_array(i+1) = PSNR(imCompressed,im);
		rate_array(i+1) =  image_rate('Mona-Lisa.jpg');
	end
	hold on
	figure()
	plot(rate_array,psnr_array, '-o')
	xlabel('Rate')
	ylabel('PSNR')
end

% =========== E: Benchmark =========== %

% =========== B: Golomb-Rice =========== %

% Tested
function codeBits = golomb_rice(num, k)
	assert(num >= 0);

	numCharArray = dec2bin(num);
	% len = length(numCharArray)-1;
	len = length(numCharArray);
	if(len > k)
		suffix = numCharArray(end-k+1:end);
		remainder = bin2dec(numCharArray(1:end-k));
		prefix = ones([1,remainder])+'0';
		codeBits = [prefix 0+'0' suffix];
	else
		suffix = zeros([1,k])+'0';
		suffix(end-len+1:end) = numCharArray;
		codeBits = [0+'0' suffix];
	end
	codeBits = cast(codeBits,'char');
end

% Tested
function num = golomb_rice_inverse(codeBits, k)
	i = 0;
	while (codeBits(i+1) == '1')
		i = i +1;
	end
	binary_val = [dec2bin(i) codeBits(end-k+1:end)];
	num = bin2dec(binary_val);
end

% Tested
function len = golomb_rice_length(num, k)
	len = length(golomb_rice(num, k));
end

% Tested
function subCodeBits = golomb_rice_find_prefix(codeBits, k)
	i = 1;
	while (codeBits(i) == '1')
		i = i +1;	
	end
	subCodeBits = codeBits(1:k+i);
end

% Tested
function golomb_rice_test()
	for index = 1:100
		clc
		a = randi([0,1000])
		k = randi([0,7])
		codeBits = golomb_rice(a,k)
		assert(golomb_rice_length(a,k) == length(codeBits));
		num = golomb_rice_inverse(codeBits, k)
		assert(isequal(a,num))
	end

	for index = 1:100
		clc
		a = randi([0,1000])
		k = randi([0,7])
		codeBits = golomb_rice(a,k)
		suffix = randi([0,1],1,5) + '0';
		extendedCodeBits = [codeBits suffix]
		subCodeBits = golomb_rice_find_prefix(extendedCodeBits,k)
		num = golomb_rice_inverse(subCodeBits, k)
		assert(isequal(a,num))
	end
end

% =========== E: Golomb-Rice =========== %
% =========== B: Exp-Golomb =========== %

% Tested
function codeBits = exp_golomb(num, k)
	assert(num >= 0);
	num = num+2.^k-1;
	numCharArray = dec2bin(num+1);
	len = length(numCharArray)-1;
	uncutCode = [(zeros(1,len)+'0') numCharArray];
	codeBits = uncutCode(k+1:end);
end

% Tested
function num = exp_golomb_inverse(codeBits, k)
	codeBitsFull = [zeros(1,k)+'0', codeBits(1:end)];
	cutIndex = find(codeBitsFull-'0', 1, 'first');
	codeBitsCut = codeBitsFull(cutIndex:end);
	num = bi2de(codeBitsCut-'0','left-msb')-1;
	num = num-2.^k+1 ;
end

% Tested
function lengthArray = exp_golomb_lengths(array, k)
	ys = arrayfun(@(x)floor(x/2.^k), array);
	lengthArray = arrayfun(@(x)2*floor(log2(double(x+1)))+1, ys) + k;
end

function subCodeBits = exp_golomb_find_prefix(codeBits, k)
	i = 0;
	while (codeBits(i+1) == '0')
		i = i + 1;
	end
	subCodeBits = codeBits(1:k+i+i+1);
end

function exp_golomb_test()
	for index = 1:100
		clc
		a = randi([0,1000])
		k = randi([0,7])
		codeBits = exp_golomb(a,k)
		assert(exp_golomb_lengths(a,k) == length(codeBits));
		c = exp_golomb_inverse(codeBits,k)
		assert(isequal(a,c));
	end

	for index = 1:100
		clc
		a = randi([0,1000])
		k = randi([0,7])
		codeBits = exp_golomb(a,k)
		suffix = randi([0,1],1,5) + '0';
		extendedCodeBits = [codeBits suffix]
		subCodeBits = exp_golomb_find_prefix(extendedCodeBits,k)
		c = exp_golomb_inverse(subCodeBits,k)
		assert(isequal(a,c));
	end
end

% =========== E: Exp-Golomb =========== %

% =========== B: Signed / Unsigned =========== %

% Tested
function val = ToUnsigned(x)
	if(x > 0)
		val = 2*x-1;
	else
		val = -2*x;
	end
end

% Tested
function val = ToSigned(x)
	if(mod(x,2))
		val = (x+1)/2;
	else
		val = x/-2;
	end
end

% =========== E: Signed / Unsigned =========== %

% =========== B: Equivalent Vector =========== %

function encoded = encode_equivalent_vector(vec,k1,k2,k3,M)
	integers = [];
	runs = [];
	last_non_zero = find(vec,1,'last');
	if(numel(last_non_zero) == 0)
		last_non_zero = 0;
	end
	fist_time = true;
	for i = 1:last_non_zero
		if(vec(i) == 0 && ~fist_time)
			runs(end) = runs(end) + 1;
		else
			fist_time = false;
			if(i ~= last_non_zero)
				runs = [runs 0];
			end
			integers = [integers vec(i)];
		end
	end

	last_non_zero_encoded = golomb_rice(last_non_zero,k1);

	runs_encoded_cells = arrayfun(@(x) golomb_rice(x,k2),runs, 'UniformOutput', false); 
	runs_encoded_cells = reshape(runs_encoded_cells, 1, numel(runs_encoded_cells));
    try
        runs_encoded = cell2mat(runs_encoded_cells);
    catch
        a = 0;
    end 
	runs_encoded = cast(runs_encoded,'char');

	integers_unsigned = arrayfun(@(x) ToUnsigned(x), integers(2:end));
	integers_encoded_cells = arrayfun(@(x) exp_golomb(x,k3), integers_unsigned, 'UniformOutput', false);
	integers_encoded_cells = reshape(integers_encoded_cells, 1, numel(integers_encoded_cells));
	integers_encoded = cell2mat(integers_encoded_cells);

	if(last_non_zero == 0)
		first_integer_bits = '0';
	else
		first_integer_bits = dec2bin(integers(1));
    end
    assert(M-length(first_integer_bits) >= 0)
	lala = zeros([1, M-length(first_integer_bits)])+'0';
	first_integer_bits = [zeros([1, M-length(first_integer_bits)])+'0' first_integer_bits];
	encoded = [last_non_zero_encoded runs_encoded first_integer_bits integers_encoded];
end

function vec = decode_equivalent_vector(encoded,N,k1,k2,k3,M)
	sub_encoded = golomb_rice_find_prefix(encoded,k1);
	last_non_zero = golomb_rice_inverse(sub_encoded,k1);
	last_non_zero_length = golomb_rice_length(last_non_zero,k1);
	encoded = encoded(last_non_zero_length+1:end);
	if(last_non_zero == 0)
		vec = zeros([1, N*N]);
		return;
	end
	runs = [];
	num_count = 1;
	while(num_count < last_non_zero)
		sub_encoded = golomb_rice_find_prefix(encoded,k2);
		val = golomb_rice_inverse(sub_encoded,k2);
		val_length = golomb_rice_length(val,k2);
		encoded = encoded(val_length+1:end);
		num_count = num_count + val + 1;
		runs = [runs val];
	end

	first_integer_bits = encoded(1:M);
	ii = find(first_integer_bits-'0',1,'first');
    first_integer = typecast(uint16(bin2dec(first_integer_bits(ii:M))), 'int16');
    if(numel(first_integer) == 0)
        first_integer = 0;
    end
	encoded = encoded(M+1:end);
	integers = zeros([1,length(runs)]);
	for i = 1:length(runs)
		sub_encoded = exp_golomb_find_prefix(encoded,k3);
		val = exp_golomb_inverse(sub_encoded,k3);
		val_length = exp_golomb_lengths(val,k3);
		encoded = encoded(val_length+1:end);
		integers(i) = val;
	end
	
	vec = zeros([1, N*N]);
	vec(1) = first_integer;
	vec_index = 1;
	for i = 1:length(runs)
		vec_index = vec_index + runs(i) + 1;
		vec(vec_index) = ToSigned(integers(i));
	end
end

function equivalent_vector_test()
	N = 8;
	% a = zeros([1,N^2]);
	a = [-228,-54,-28,-8,22,4,-15,-9,9,1,-7,-18,-10,-6,0,2,9,-10,2,7,0,-1,1,-15,-3,2,1,18,1,-1,-1,-20,0,-1,-1,0,0,0,1,-2,1,-2,-20,0,1,0,1,-1,-1,0,0,0,-2,0,0,1,0,1,0,0,0,0,0,0];
	k1 = 3;
	k2 = 2;
	k3 = 0;
	M = 16;
	encoded = encode_equivalent_vector(a,k1,k2,k3,M);
	decoded = decode_equivalent_vector(encoded,N,k1,k2,k3,M);
	assert(isequal(decoded, a));
end

% =========== E: Equivalent Vector =========== %

% =========== B: 4 =========== %

function Q4()
	N = 8;
	k1 = 3;
	k2 = 2;
	k3 = 0;
	b = 1;
	im = imread('Mona-Lisa.bmp');
	count = 20;
	psnr_array = zeros([1,count]);
	b_array = zeros([1,count]);
	for i = 1:count
		current_b = b*2^(i-1);
		new_im = compress_im_Q4(im,N,k1,k2,k3,current_b);
		psnr_array(i) = PSNR(new_im, im);
		b_array(i) = current_b;
	end

	hold on
	figure()
	semilogx(b_array, psnr_array, '-o');
end

function compressed_im = compress_im_Q4(im,N,k1,k2,k3,b)
	normalized_im = arrayfun(@(x) normalize(x),im);
	structMat = splitMat2Struct(normalized_im, N);
	for i = 1:size(structMat,1)
		for j = 1:size(structMat,2)
			submat = structMat(i,j).submat;
			dct_vals = dct2(submat);
			dct_vals = round(dct_vals);
			dct_quantized = round(dct_vals./b);
			zigzag_array = zigzag(dct_quantized);
			new_sct_vals = izigzag(zigzag_array, size(submat));
			new_sct_vals = new_sct_vals.*b;
			new_submat = idct2(new_sct_vals);
			structMat(i,j).submat = new_submat;
		end
	end

	mat = structMat2Mat(structMat);
	mat(mat>127) = 127;
	mat(mat<-128) = -128;
	compressed_im = arrayfun(@(x) denormalize(x), mat);

	% imshow(im);
	% figure()
	% imshow(denormalized_im);
end

% =========== E: 4 =========== %

% =========== B: 6 =========== %

function Q6()
	N = 8;
	k1 = 3;
	k2 = 2;
	k3 = 0;
	b = 0.1;
	count = 25;%25;%48;
	M = 8;
	im = imread('Mona-Lisa.bmp');
	mse_array = zeros([1,count]);
	psnr_array = zeros([1,count]);
	rate_array = zeros([1,count]);
	b_array=linspace(0.05,0.99,count);
	figure;
	subplot(ceil(sqrt(count)),ceil(sqrt(count)),1);
	imshow(im);
	title('Q6 - Original Image');
	for i = 1:count
		disp(['Q6:' num2str(i) '/' num2str(count)])
		current_b = b_array(i);
		subplot(ceil(sqrt(count+1)),ceil(sqrt(count+1)),i+1);
		[mse_val, psnr_val, rate] = compress_im_Q6(im,N,k1,k2,k3,current_b,M);
		title(['b=' num2str(current_b)]);
		mse_array(i) = mse_val;
		psnr_array(i) = psnr_val;
		rate_array(i) = rate;
		b_array(i) = current_b;
	end

	figure()
	subplot(1,2,1);
	plot(rate_array, psnr_array, '-o');
	xlabel('Rate')
	ylabel('PSNR')
	title('Q6 - PSNR')
	subplot(1,2,2);
	plot(rate_array, mse_array, '-o');
	xlabel('Rate')
	ylabel('MSE')
	title('Q6')
	figure
	plot(b_array, rate_array, '-o');
	xlabel('b')
	ylabel('Rate')
	title('Q6 - b/rate')
end

function [mse_val, psnr_val, rate] = compress_im_Q6(im,N,k1,k2,k3,b,M)
	normalized_im = arrayfun(@(x) normalize(x),im);
	structMat = splitMat2Struct(normalized_im, N);
	encodeded_size = 0;
	rations = [];
	for i = 1:size(structMat,1)
		for j = 1:size(structMat,2)
			submat = structMat(i,j).submat;
			dct_vals = dct2(submat);
			dct_quantized = round(dct_vals./b);
			zigzag_array = zigzag(dct_quantized);
			encodeded = encode_equivalent_vector(zigzag_array,k1,k2,k3,M);
			rations = [rations length(encodeded)/(size(submat,1)*size(submat,2))];
			encodeded_size = encodeded_size + length(encodeded);
			decoded = decode_equivalent_vector(encodeded,N,k1,k2,k3,M);
			new_sct_vals = izigzag(decoded, size(submat));
			new_sct_vals = new_sct_vals.*b;
			new_submat = idct2(new_sct_vals);
			structMat(i,j).submat = new_submat;
		end
	end

	num_pix = size(im,1)*size(im,2);
	rate = encodeded_size/(8*num_pix);
	mat = structMat2Mat(structMat);
	% mat(mat>127) = 127;
	% mat(mat<-128) = -128;
	compressed_im = arrayfun(@(x) denormalize(x), mat);
	mse_val = MSE(compressed_im, im);
	psnr_val = PSNR(compressed_im, im);
	imshow(compressed_im);
end

% =========== E: 6 =========== %
% =========== B: 7 =========== %

function Q7()
	N = 8;
	k1 = 3;
	k2 = 2;
	k3 = 0;
	b = 0.1;
	count = 48;
	M = 16;
	im = imread('Mona-Lisa.bmp');
	mse_array = zeros([1,count]);
	psnr_array = zeros([1,count]);
	rate_array = zeros([1,count]);
	b_array=linspace(0.05,0.99,count);
	figure;
	subplot(ceil(sqrt(count)),ceil(sqrt(count)),1);
	imshow(im);
	title('Q7 - Original Image');
	% imshow(compressed_im);
	for i = 1:count
		disp(['Q7:' num2str(i) '/' num2str(count)])
		current_b = b_array(i);
		subplot(ceil(sqrt(count+1)),ceil(sqrt(count+1)),i+1);
		[mse_val, psnr_val, rate] = compress_im_Q7(im,N,k1,k2,k3,current_b,M);
		title(['b=' num2str(current_b)]);
		mse_array(i) = mse_val;
		psnr_array(i) = psnr_val
		rate_array(i) = rate;
	end

	figure()
	subplot(1,2,1);
	plot(rate_array, psnr_array, '-o');
	xlabel('Rate')
	ylabel('PSNR')
	title('Q7')
	subplot(1,2,2);
	plot(rate_array, mse_array, '-o');
	xlabel('Rate')
	ylabel('MSE')
	title('Q7')
	figure
	plot(b_array, rate_array, '-o');
	xlabel('b')
	ylabel('Rate')
	title('Q7 - b/rate')
end

function [mse_val, rate] = compress_im_Q7(im,N,k1,k2,k3,b,M)
	normalized_im = arrayfun(@(x) normalize(x),im);
	structMat = splitMat2Struct(normalized_im, N);
	encodeded_size = 0;
	last_dc = 0;
	for i = 1:size(structMat,1)
		for j = 1:size(structMat,2)
			submat = structMat(i,j).submat;
			dct_vals = dct2(submat);
			if(i ~= 1 || j ~= 1)
				current_dc = dct_vals(1,1);
				dct_vals(1,1) = current_dc - last_dc;
				last_dc = current_dc;
			end
			dct_quantized = round(dct_vals./b);
			zigzag_array = zigzag(dct_quantized);
			encodeded = encode_equivalent_vector(zigzag_array,k1,k2,k3,M);
			encodeded_size = encodeded_size + length(encodeded);
			decoded = decode_equivalent_vector(encodeded,N,k1,k2,k3,M);
			new_sct_vals = izigzag(decoded, size(submat));
			new_sct_vals = new_sct_vals.*b;
			if(i ~= 1 || j ~= 1)
				new_sct_vals(1,1) = new_sct_vals(1,1) + last_dc;
			end
			new_submat = idct2(new_sct_vals);
			structMat(i,j).submat = new_submat;
		end
	end

	num_pix = size(im,1)*size(im,2);
	rate = encodeded_size/(8*num_pix);
	mat = structMat2Mat(structMat);
	% mat(mat>127) = 127;
	% mat(mat<-128) = -128;
	compressed_im = arrayfun(@(x) denormalize(x), mat);
	mse_val = MSE(compressed_im, im);

	imshow(compressed_im);
end

% =========== E: 7 =========== %


function test_dct()
	im = imread('Mona-Lisa.bmp');
	dct_vals = dct2(im);
end
