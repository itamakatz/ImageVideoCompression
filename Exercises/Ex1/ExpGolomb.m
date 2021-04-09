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
binPath = 'encodeIm.bin';
q12_encode(im, binPath, 4, 2);
im2 = q12_dencode(binPath);
im2 = cast(im2,'uint8');
imwrite(im2,'decoded.bmp');
imshow(im2);
lala = 0;

% Tested
function im = q12_dencode(binPath)
	fileID = fopen(binPath,'r');

	k = cast(fread(fileID, 1, '*ubit8', 'ieee-be'),'uint16');
	n = cast(fread(fileID, 1, '*ubit8', 'ieee-be'),'uint16');
	imDim1 = fread(fileID, 1, '*ubit16', 'ieee-be');
	imDim2 = fread(fileID, 1, '*ubit16', 'ieee-be');
	

	subDim1 = imDim1/n;
	subDim2 = imDim2/n;
	block_pixels = n*n;

	im = zeros(imDim1, imDim2);

	% parsing the file into a struct array containing all the number encodings
	for i = 1:subDim1
		for j = 1:subDim2
			m = fread(fileID, 1, '*ubit8', 'ieee-be');
			m = cast(m,'int16');
			pixelCount = 0;
			blockPixels = zeros(1,n*n);
			while (pixelCount < block_pixels)
				prefix_zeros = 0;
				while(fread(fileID, 1, '*ubit1', 'ieee-be') == 0)
					prefix_zeros = prefix_zeros + 1;
				end
				bits = [zeros(1,prefix_zeros)+'0' '1' char(fread(fileID, prefix_zeros + k, '*ubit1', 'ieee-be')+'0')'];
				val = dencode_number(bits, k);
				signedVal = ToSigned(cast(val,'int16'));
				originalVal = signedVal + m;
				blockPixels(pixelCount+1) = originalVal;
				pixelCount = pixelCount + 1;
			end
			
			im(n*(i-1)+1:n*(i), n*(j-1)+1:n*(j)) = reshape(blockPixels,[n, n])'; %note the tranpose
		end
	end

	fclose(fileID);
end

% Tested
function q12_encode(im, binPath, n, k)
	structMat = splitMat2Struct(im, n);
	fileID = fopen(binPath,'w');
	fwrite(fileID, k, 'ubit8', 'ieee-be');
	fwrite(fileID, n, 'ubit8', 'ieee-be');
	fwrite(fileID, size(im,1), 'ubit16', 'ieee-be');
	fwrite(fileID, size(im,2), 'ubit16', 'ieee-be');
	for i = 1:size(structMat,1)
		for j = 1:size(structMat,2)
			submat = structMat(i,j).submat;
			submat = cast(submat,'int16');
			m = round(mean(submat,'all'));
			mDiff = submat - m;
			unsignedVals = arrayfun(@(x)ToUnsigned(x), mDiff)'; % note the transpose of the matrix!
			charCellsVals = arrayfun(@(x)encode_number(x, k), unsignedVals, 'UniformOutput', false);
			charCellsVals = reshape(charCellsVals, 1, numel(charCellsVals));
			charArrayVals = cell2mat(charCellsVals);
			fwrite(fileID, m, 'ubit8', 'ieee-be');
			fwrite(fileID, charArrayVals-'0', 'ubit1', 'ieee-be');
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

% Tested
function num = dencode_number(codeBits, k)
	codeBitsFull = [zeros(1,k)+'0' codeBits];
	num = dencode_number_zero_based(codeBitsFull)-2.^k+1 ;
end

% Tested
function num = dencode_number_zero_based(codeBits)
	cutIndex = find(codeBits-'0', 1, 'first');
	codeBitsCut = codeBits(cutIndex:end);
	num = bi2de(codeBitsCut-'0','left-msb')-1;
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


% == TESTS == 
% pairs = [struct('a',66,'k',1), 
% 		 struct('a',180,'k',2),
% 		 struct('a',15,'k',1),
% 		 struct('a',999,'k',7)];

% arrayfun(@(x)(dencode_number(encode_number(x.a,x.k),x.k) == x.a), pairs)