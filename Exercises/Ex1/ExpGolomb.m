clc
clear variables
close all

decArray = [0 8 17;
			5 9 27];

%k0 = expGolombLengths(decArray, 0);
%k1 = expGolombLengths(decArray, 1);
%k2 = expGolombLengths(decArray, 2);

im = imread('Mona-Lisa.bmp');

q10(im, 4, 0)

function len = q10(im, n, k)
	len = 0;
	structMat = splitMat2Struct(im, n);
	structMatSize = size(structMat);
	for i = 1:structMatSize(1)
		for j = 1:structMatSize(2)
			submat = structMat(i,j).submat;
			submat = cast(submat,'int8');
			m = mean(submat,'all');
			mDiff = submat - m;
			unsignedVals = arrayfun(@(x)ToUnsigned(x), mDiff);
			len = len + expGolombLengths(unsignedVals, k) + 8;
		end
	end
end

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