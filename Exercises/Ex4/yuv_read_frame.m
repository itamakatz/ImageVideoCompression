%-----------------------------------------------------------------------------
% Read a single frame from .yuv or .y4m file
%-----------------------------------------------------------------------------
function frame = yuv_read_frame(video_file_name, width, height, frame_idx, bits_per_color, chroma_sampling)

% Check if file format is .yuv or .y4m
switch(video_file_name(end-2:end))
    case 'yuv'
        y4m_file = false;
        
    case 'y4m'
        y4m_file = true;
        
        % fine the beginning of data pixels
        byte_for_parser = 200;
        fid = fopen(video_file_name, 'rb');
        x = fread(fid, byte_for_parser, 'uint8');
        fclose(fid);
        
        if(strcmp(char(x(1:10)'), 'YUV4MPEG2 '))
            p_frame = strfind(char(x'), 'FRAME');
        else
            error('File does not contain the Y4M expected header.\n');
        end

    otherwise
        error('File format not supported.');
end


switch(bits_per_color)
    case 8
        bpc_str = 'uint8';
        bytes_per_color = 1;
    case {10, 12}
        bpc_str = 'uint16';
        bytes_per_color = 2;
    otherwise
        error('Bits per Color should be 8, 10 or 12!\n');
end

switch(chroma_sampling)
    case '420'
        L = 1.5 * width * height;  % Length of a single frame (including: luma, chroma)
        
        fid = fopen(video_file_name, 'rb');
        if(y4m_file)
            fseek(fid, p_frame-1 + 6 + (L+6)*(frame_idx-1)*bytes_per_color, 'bof');
        else
            fseek(fid,                 (L  )*(frame_idx-1)*bytes_per_color, 'bof');
        end
        x = fread(fid, L, bpc_str);
        fclose(fid);
        
        Ly = width * height;
        Lc = Ly / 4;
        y = x(1:Ly);
        u = x(Ly + (1:Lc));
        v = x(Ly + Lc + (1:Lc));
        
        y = reshape(y, width   , height  )';
        u = reshape(u, width/2 , height/2)';
        v = reshape(v, width/2 , height/2)';
        
    case '444'
        L = 3 * width * height;  % Length of a single frame (including: luma, chroma)
        
        fid = fopen(video_file_name, 'rb');
        if(y4m_file)
            fseek(fid, p_frame-1 + 6 + (L+6)*(frame_idx-1), 'bof');
        else
            fseek(fid,                 (L  )*(frame_idx-1), 'bof');
        end
        x = fread(fid, L, bpc_str);
        fclose(fid);
        
        Ly = width * height;
        Lc = Ly;
        y = x(1:Ly);
        u = x(Ly + (1:Lc));
        v = x(Ly + Lc + (1:Lc));
        
        y = reshape(y, width , height)';
        u = reshape(u, width , height)';
        v = reshape(v, width , height)';

    otherwise
        error('Chroma sampling should be 420 or 444.\n');
        
end

frame.y = y;
frame.cr = u;
frame.cb = v;
    