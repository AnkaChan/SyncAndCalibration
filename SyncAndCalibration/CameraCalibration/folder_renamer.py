import os


if __name__ == '__main__':
	root_dir = r'\\DATACHEWER\shareZ\2019_12_13_Lada_Capture\Converted'
	cams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
	for cam_idx in range(16):
		from_dir = root_dir + '\\' + cams[cam_idx] + '\\Corners_20191214'
		to_dir = root_dir + '\\' + cams[cam_idx] + '\\Corners'
		os.rename(from_dir, to_dir)
		print('Camera {}: {} -> {}'.format(cams[cam_idx], from_dir, to_dir))

	print('* Complete!')
