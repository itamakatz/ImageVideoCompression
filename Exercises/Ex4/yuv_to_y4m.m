
width           = 1920; % [pixels] - Frame Width
height          = 1080; % [pixels] - Frame Height
start_frame         =  201;
end_frame         =  220; %300
bits_per_color  =    8; % 
chroma_sampling ='420';


input_file_name         = 'ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv';
output_file_name        = 'short_bw.y4m';

out_fid = fopen(output_file_name, 'wb');

y4m_header = uint8(['YUV4MPEG2 W1920 H1080 F60:1 Ip A0:0 C420jpeg XYSCSS=420JPEG' 10]);
fwrite(out_fid, y4m_header, 'uint8');

count = 1;
for frame_idx = start_frame : 1 : end_frame
% for frame_idx = start_frame : end_frame
% for frame_idx = end_frame : -5 : start_frame 

    frame = yuv_read_frame(input_file_name, width, height, frame_idx, bits_per_color, chroma_sampling);
    frame.cr = frame.cr*0 + 128;
    frame.cb = frame.cb*0 + 128;
    write_yuv_frame(out_fid, frame);
    count = count + 1;
end

fclose(out_fid);

disp(count)
disp('finished')