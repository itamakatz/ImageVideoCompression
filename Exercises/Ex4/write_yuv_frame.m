
function [] = write_yuv_frame(fid, frame)

fwrite(fid, ['FRAME' 10], 'uint8');
fwrite(fid, reshape((frame.y)', [], 1), 'uint8');
fwrite(fid, reshape((frame.cr)', [], 1), 'uint8');
fwrite(fid, reshape((frame.cb)', [], 1), 'uint8');
