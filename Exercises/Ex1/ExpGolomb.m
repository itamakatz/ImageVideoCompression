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
% q12_encode(im, 4, 2)
k=2
lenPractical = arrayfun(@(x) length(encode_number(x,k)),decArray)
lenTheoretical = expGolombLengths(decArray,k)
isequal(lenPractical, lenTheoretical)
% encode_number_zero_based(8+2*2-1)

function im = q12_dencode(binPath)

end

function q12_encode(im, binPath, n, k)
	structMat = splitMat2Struct(im, n);
	fileID = fopen(binPath,'w');
	for i = 1:size(structMat,1)
		for j = 1:size(structMat,2)
			submat = structMat(i,j).submat;
			submat = cast(submat,'int16');
			m = mean(submat,'all');
			fwrite(fileID, m, 'ubit8')

		end
	end
	fclose(fileID)
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

% Best is n=4, k=2
function optimal = optimize_q11(im)
	optimal = struct("n",-1,"k",-1,"meanLength",-1);
	for n = [4,8,16,24,48]
		for k = 0:7
			lengths = q11(im, n, k);
			meanLength = sum(lengths,'all');
			disp(['n: ' num2str(n) ', k: ' num2str(k), ', meanLength: ' num2str(meanLength)])
			if(optimal.meanLength == -1 || optimal.meanLength > meanLength)
				optimal = struct("n", n, "k", k, "meanLength", meanLength);
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
			m = mean(submat,'all');
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