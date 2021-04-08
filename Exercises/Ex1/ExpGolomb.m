% clc
% clear variables
% close all

decArray = [0 8 17;
			5 9 27];

%k0 = expGolombLengths(decArray, 0);
%k1 = expGolombLengths(decArray, 1);
%k2 = expGolombLengths(decArray, 2);

im = imread('Mona-Lisa.bmp');

% q11(im, 4, 0)
% q11(im, 16, 0)
% optimize_q11(im)
binPath = 'encodeIm.bin'
% q12_encode(im, binPath, 4, 2);
% q12_dencode(binPath)
encode_number_zero_based(7)
function im = q12_dencode(binPath)
	fileID = fopen(binPath,'r');
	k = fread(fileID, 1, '*ubit8');
	n = fread(fileID, 1, '*ubit8');
	imDim1 = fread(fileID, 1, '*ubit16');
	imDim2 = fread(fileID, 1, '*ubit16');

	subDim1 = imDim1/n;
	subDim2 = imDim2/n;

	block_pixels = subDim1*subDim2;

	blocks = cell(n,n);

	for i = 1:n
		for j = 1:n
			blocks(i,j)	= struct('mean', -1, 'pixels', cell(subDim1, subDim2));
			m = fread(fileID, 1, '*ubit8');
			blocks(i,j).mean = m
			count = 0
			while (count < block_pixels)
				bits_to_read = k;
				prefix_zeros = 0;
				bits = []
				while(fread(fileID, 1, '*ubit1') == 0)
					bits = [bits 0];
					prefix_zeros = prefix_zeros + 1;
				end
				bits = [bits 1];
				fread(fileID, 1, '*ubit1')
				for i = range(prefix_zeros)
					
				end
				count = count + 1;
			end

		end
	end
	while ~feof(fileID)
		tline = fgetl(fid);
		disp(tline)
	end
	fclose(fileID);
end

% Tested
function q12_encode(im, binPath, n, k)
	structMat = splitMat2Struct(im, n);
	fileID = fopen(binPath,'w');
	fwrite(fileID, k, 'ubit8');
	fwrite(fileID, n, 'ubit8');
	fwrite(fileID, size(structMat,1), 'ubit16');
	fwrite(fileID, size(structMat,2), 'ubit16');
	for i = 1:size(structMat,1)
		for j = 1:size(structMat,2)
			submat = structMat(i,j).submat;
			submat = cast(submat,'int16');
			m = round(mean(submat,'all'));
			mDiff = submat - m;
			unsignedVals = arrayfun(@(x)ToUnsigned(x), mDiff);
			charCellsVals = arrayfun(@(x)encode_number(x, k), unsignedVals, 'UniformOutput', false);
			charCellsVals = reshape(charCellsVals', 1, numel(charCellsVals)); % note the transpose of the matrix
			charArrayVals = cell2mat(charCellsVals);
			fwrite(fileID, m, 'ubit1');
			fwrite(fileID, charArrayVals, 'ubit1');
		end
	end
	fclose(fileID);
end

% Tested
function codeBits = encode_number(num, k)
	uncutCode = encode_number_zero_based(num+2.^k-1);
	codeBits = uncutCode(k+1:end);
end

% Tested
function codeBits = encode_number_zero_based(num)
	numCharArray = dec2bin(num+1);
	len = length(numCharArray)-1;
	codeBits = [(zeros(1,len)+'0') numCharArray];
end

function codeBits = dencode_number(num, k)
	uncutCode = encode_number_zero_based(num+2.^k-1);
	codeBits = uncutCode(k+1:end);
end

function codeBits = dencode_number_zero_based(num)
	numCharArray = dec2bin(num+1);
	len = length(numCharArray)-1;
	codeBits = [(zeros(1,len)+'0') numCharArray];
end

% Best is n=4, k=2, totalLength: 3813686 bits = 476710.75 bytes
function optimal = optimize_q11(im)
	optimal = struct("n",-1,"k",-1,"totalLength",-1);
	for n = [4,8,16,24,48]
		for k = 0:7
			lengths = q11(im, n, k);
			totalLength = sum(lengths,'all');
			disp(['n: ' num2str(n) ', k: ' num2str(k), ', totalLength: ' num2str(totalLength)])
			if(optimal.totalLength == -1 || optimal.totalLength > totalLength)
				optimal = struct("n", n, "k", k, "totalLength", totalLength);
			end
		end
	end
end

function len = q11(im, n, k)
	len = 0;
	structMat = splitMat2Struct(im, n);
	for i = 1:size(structMat,1)
		for j = 1:size(structMat,2)
			submat = structMat(i,j).submat;
			submat = cast(submat,'int16');
			m = round(mean(submat,'all'));
			mDiff = submat - m;
			unsignedVals = arrayfun(@(x)ToUnsigned(x), mDiff);
			len = len + expGolombLengths(unsignedVals, k) + 8;
		end
	end
end

% Tested
function lengthArray = expGolombLengths(array, k)
	% unsigned = arrayfun(@(x)ToUnsigned(x), array)
	ys = arrayfun(@(x)floor(x/2.^k), array);
	lengthArray = arrayfun(@(x)2*floor(log2(double(x+1)))+1, ys) + k;
end

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