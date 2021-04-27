a = [ 25 0 0 -13 11 0 7 0 0 0 0 ]
k1 = 2
k2 = 2
k3 = 2
encode_equivalent_vector(a, k1,k2,k3)
% benchmark()
% quantization(randi([-100, 100],1,10)./200, 0.02)
% golomb_rice_test()
% exp_golomb_test()
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
	new_val = val/256 - 0.5;
end

% Tested
function new_val = denormalize(val)
	assert(val >= -0.5 && val < 0.5)
	new_val = (val + 0.5)*256;
end

% Tested
function success = normalize_test()
	val = randi([0,255]);
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
		% bmp_rate = image_rate('Mona-Lisa.bmp')
		rate_array(i+1) =  image_rate('Mona-Lisa.jpg');
	end
	plot(rate_array,psnr_array)
	xlabel('Rate')
	ylabel('PSNR')
end

% =========== E: Benchmark =========== %
% =========== B: 4 =========== %

% =========== E: 4 =========== %

% =========== B: Equivalent Vector =========== %

function encoded = encode_equivalent_vector(vec,k1,k2,k3)
	integers = [];
	runs = [];
	last_zero = find(vec,1,'last');
	fist_time = true;
	for i = 1:last_zero
		if(vec(i) == 0 && ~fist_time)
			runs(end) = runs(end) + 1;
		else
			fist_time = false;
			runs = [runs 0];
			integers = [integers vec(i)];
		end
	end

	assert(runs(end) == 0);
	runs = runs(1:end-1);
	assert(runs(end) ~= 0);

	last_zero_encoded = golomb_rice(last_zero,k1);

	integers_unsigned =  arrayfun(@(x) ToUnsigned(x), integers);
	integers_encoded_cells = arrayfun(@(x) golomb_rice(x,k2),integers_unsigned, 'UniformOutput', false);
	integers_encoded_cells = reshape(integers_encoded_cells, 1, numel(integers_encoded_cells));
	integers_encoded = cell2mat(integers_encoded_cells);

	runs_encoded_cells = arrayfun(@(x) golomb_rice(x,k3),runs, 'UniformOutput', false);
	runs_encoded_cells = reshape(runs_encoded_cells, 1, numel(runs_encoded_cells));
	runs_encoded = cell2mat(runs_encoded_cells);

	encoded = [last_zero_encoded integers_encoded runs_encoded];
end

function vec = deencode_equivalent_vector(encoded,k1,k2,k3)

	index = 1;
	last_zero = golomb_rice_inverse(encoded,k1);
	last_zero_length = golomb_rice_length(last_zero,k1);
	index = index + last_zero_length;
	
	runs_encoded_cells = arrayfun(@(x) golomb_rice(x,k2),runs, 'UniformOutput', false);
	runs_encoded_cells = reshape(runs_encoded_cells, 1, numel(runs_encoded_cells));
	runs_encoded = cell2mat(runs_encoded_cells);

	integers_unsigned =  arrayfun(@(x) ToUnsigned(x), integers);
	integers_encoded_cells = arrayfun(@(x) exp_golomb(x,k3),integers_unsigned(2:end), 'UniformOutput', false);
	integers_encoded_cells = reshape(integers_encoded_cells, 1, numel(integers_encoded_cells));
	integers_encoded = cell2mat(integers_encoded_cells);
	
	encoded = [last_zero_encoded runs_encoded integers_encoded];
end

% =========== E: Equivalent Vector =========== %

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
		prefix = zeros([1,remainder])+'0';
		codeBits = [prefix 1+'0' suffix];
	else
		suffix = zeros([1,k])+'0';
		suffix(end-len+1:end) = numCharArray;
		codeBits = [1+'0' suffix];
	end
end

% Tested
function num = golomb_rice_inverse(codeBits, k)
	i = 0;
	while (codeBits(i+1) == '0')
		i = i +1;
	end
	binary_val = [dec2bin(i) codeBits(end-k+1:end)];
	num = bin2dec(binary_val);
end

function len = golomb_rice_length(num, k)
	len = length(golomb_rice(num, k));
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
end

% =========== E: Golomb-Rice =========== %
% =========== B: Exp-Golomb =========== %

% Tested
function codeBits = exp_golomb(num, k)
	% accepts negative numbers and simply adds a sign bit at the end. 0 is positive, 1 negative
	if(num > 0)
		sign_bit = 0 +'0';
	else
		sign_bit = 1 +'0';
	end
	num = abs(num);
	% assert(num >= 0);
	num = num+2.^k-1;
	numCharArray = dec2bin(num+1);
	len = length(numCharArray)-1;
	uncutCode = [(zeros(1,len)+'0') numCharArray];
	codeBits = uncutCode(k+1:end);
	codeBits = [codeBits sign_bit];
end

% Tested
function num = exp_golomb_inverse(codeBits, k)
	% can return a negative number
	codeBitsFull = [zeros(1,k)+'0', codeBits(1:end-1)];
	cutIndex = find(codeBitsFull-'0', 1, 'first');
	codeBitsCut = codeBitsFull(cutIndex:end);
	num = bi2de(codeBitsCut-'0','left-msb')-1;
	num = num-2.^k+1 ;
	if(codeBits(end) == '1')
		num = num * -1;
	end
end

% Tested
function lengthArray = exp_golomb_lengths(array, k)
	% length for the code + 1 sign bit
	array = abs(array) % calculate length for the abs values and add one at the end
	ys = arrayfun(@(x)floor(x/2.^k), array);
	lengthArray = arrayfun(@(x)2*floor(log2(double(x+1)))+1, ys) + k + 1; % the last + 1 is for the sign bit
end

function exp_golomb_test()
	for index = 1:100
		clc
		a = randi([-1000,1000])
		k = randi([0,7])
		codeBits = exp_golomb(a,k)
		negative_codeBits = exp_golomb(a*-1,k)
		assert(isequal(codeBits(1:end-1), negative_codeBits(1:end-1)));
		assert(~isequal(codeBits(end), negative_codeBits(end)));
		assert(exp_golomb_lengths(a,k) == length(codeBits));
		c = exp_golomb_inverse(codeBits,k)
		assert(isequal(a,c));
	end
end

% =========== E: Exp-Golomb =========== %
% =========== B: Unifrom Quantization =========== %

function mat = quantization(mat, b)
	mat = mat./b
	minVal = min(abs(mat));
	if(minVal < 1)
		mat = round(mat / minVal)*minVal;
	else
		mat = round(mat);
	end
end

% =========== E: Unifrom Quantization =========== %

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