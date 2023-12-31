import random

location_list = ['hành lang',
                 'hầm rượu',
                 'nhà tắm',
                 'phòng chờ',
                 'phòng giặt',
                 'phòng đợi',
                 'WC',
                 'ban công',
                 'ban công số 3',
                 'bếp',
                 'bếp số 2',
                 'cầu thang',
                 'cầu thang của Quân',
                 'cổng',
                 'cổng chính',
                 'garage',
                 'gác xép',
                 'hiên',
                 'hiên sau',
                 'hiên trước',
                 'hành lang',
                 'hành lang Trường Sa',
                 'hè',
                 'hè sau',
                 'hè trước',
                 'hầm rượu',
                 'hầm rượu Mùa Xuân',
                 'hầm rượu Mùa xuân',
                 'hầm rượu mùa xuân',
                 'nhà bếp',
                 'nhà bếp số 8',
                 'nhà giữ đồ',
                 'nhà giữ đồ Trường Sa',
                 'nhà giữ đồ của Vy',
                 'nhà giữ đồ trường sa',
                 'nhà tắm',
                 'nhà vệ sinh',
                 'nhà vệ sinh của Đạt',
                 'nhà xe Tình yêu',
                 'nhà xe của Huy',
                 'nhà xe tình yêu',
                 'nhà để xe',
                 'nhà để xe của Duy',
                 'phòng',
                 'phòng bếp',
                 'phòng chung',
                 'phòng chung của em bé',
                 'phòng chính',
                 'phòng chơi game',
                 'phòng chờ',
                 'phòng chờ số 4',
                 'phòng cuối',
                 'phòng của Long',
                 'phòng của long',
                 'phòng giặt',
                 'phòng giặt là bên trái',
                 'phòng giặt là của Hân',
                 'phòng giặt là số 5',
                 'phòng giặt ủi của Nhi',
                 'phòng học',
                 'phòng họp',
                 'phòng họp gia đình',
                 'phòng khách',
                 'phòng khách của ông bà',
                 'phòng làm việc',
                 'phòng nghỉ',
                 'phòng nghỉ của Nhi',
                 'phòng nghỉ của nhi',
                 'phòng nghỉ ngơi',
                 'phòng ngủ',
                 'phòng ngủ của My',
                 'phòng sinh hoạt chung',
                 'phòng sách số 8',
                 'phòng thay đồ số 2',
                 'phòng thay đồ số 4',
                 'phòng thu',
                 'phòng thờ',
                 'phòng tiếp đón',
                 'phòng tiếp đón của Linh',
                 'phòng tiếp khách',
                 'phòng truyền thống Trường Sa',
                 'phòng tập gym',
                 'phòng tập gym của My',
                 'phòng tắm',
                 'phòng tắm của ông bà',
                 'phòng tắm xông hơi',
                 'phòng tắm xông hơi bên phải',
                 'phòng vệ sinh',
                 'phòng vệ sinh Hoàng Hôn',
                 'phòng vệ sinh số 9',
                 'phòng xem phim',
                 'phòng xem phim của Hân',
                 'phòng xông hơi bên trái',
                 'phòng xông hơi số 1',
                 'phòng đọc',
                 'phòng đọc của Khôi',
                 'phòng đọc phía đông',
                 'phòng đợi',
                 'sân',
                 'sân sau',
                 'sân số 7',
                 'sân thượng',
                 'sân trước',
                 'sảnh',
                 'thư viện',
                 'tiền sảnh',
                 'tầng hầm',
                 'tầng thượng',
                 'vườn',
                 'wc',
                 'đại sảnh',
                 'đại sảnh bên phải',
                 'đầu hè',
                 'phòng tắm',
                 'phòng lễ tân',
                 'sảnh chính',
                 'tầng 1',
                 'tầng 2',
                 'tầng 3',
                 'phòng ăn',
                 'phòng khách tầng 1',
                 'phòng ngủ tầng 2',
                 'phòng ăn tầng 3',
                 "phòng tắm", "phòng khách", "phòng ngủ", "nhà vệ sinh", "nhà bếp", "vườn", "sân",
                 "phòng áp mái", "nhà để xe", "phòng sưởi nắng", "nhà kho", "phòng vệ sinh",
                 "phòng chung", "phòng nghỉ", "phòng sinh hoạt chung", "phòng giặt", "phòng giặt ủi",
                 "phòng đọc", "phòng sách", "phòng đọc sách", "phòng sinh hoạt chung vũ trụ",
                 "phòng chơi game", "phòng xem phim", "phòng thờ", "phòng thay đồ", "phòng truyền thống",
                 "phòng truyền thống hướng đông", "phòng tắm", "phòng xông hơi", "phòng tắm xông hơi",
                 "phòng giặt là hoàng hôn", "phòng giặt là", "phòng bếp", "phòng khách hoa sen",
                 "phòng chơi game sáng tạo", "phòng nghỉ ngơi", "phòng học", "phòng chứa đồ",
                 "phòng giải trí", "sân sau", "hiên", "sân trước", "tầng hầm", "gác xép", "gác",
                 "garage", "phòng gia đình", "phòng ăn", "phòng để đồ", "phòng kho", "kho", "phòng họp",
                 "ban công", "phòng tắm nắng", "sảnh", "hành lang", "phòng giải lao", "tiền sảnh",
                 "phòng spa", "phòng để đồ vải", "phòng thiền", "phòng billiard", "phòng bi-a",
                 "phòng chơi điện tử", "phòng playstation", "phòng spa tại gia", "phòng âm thanh",
                 "phòng âm nhạc", "phòng truyền thông đa phương tiện", "phòng vẽ", "phòng nghệ thuật",
                 "phòng làm việc", "văn phòng", "thư viện", "phòng rượu", "hầm rượu", "phòng rượu vang",
                 "kho rượu", "quầy bar", "phòng thú cưng", "phòng cho thú cưng", "phòng an toàn",
                 "phòng thiên văn", "phòng quan sát", "đài quan sát", "phòng tiện ích"
                 ]

synonym = {"nhớ": ["nha", "nhe", ""],
           "nhá": ["nha", "nhe", ""],
           "nhé": ["nhớ", "nhe", ""],
           "quá": ["thật đấy", "nhỉ", "nhở"],
           "ơi": ["này", "", 'à'],
           "rồi": ["rùi", ""],
           "với": [""],
           "giúp": ["hộ", "dùm", "cho"],
           "cho": ["hộ", "dùm"],
           "Giúp": ["Hộ", "Dùm"],
           "tau": ["tao"],
           "cái": [""]
           }

possible_intent_device_mapping = {'bật thiết bị': ['bình nóng lạnh',
                                                   'bóng chùm', 'bóng compact', 'bóng huỳnh quang', 'bóng hắt',
                                                   'bóng ngủ',
                                                   'bóng đèn chùm', 'bóng đèn compact', 'bóng đèn huỳnh quang',
                                                   'bóng đèn hắt',
                                                   'bóng đèn ngủ', 'bóng đèn sưởi', 'bóng đèn treo tường',
                                                   'bóng đèn tròn',
                                                   'bóng đèn tuýp', 'bóng đèn đứng', 'bóng đèn ốp trần',
                                                   'bếp', 'bếp điện', 'bếp ga', 'camera', 'camera giám sát'
                                                                                          'laptop', 'PC', 'loa',
                                                   'loa ti vi', 'loa máy tính', 'lò nướng', 'lò vi sóng',
                                                   'lò sưởi',
                                                   'máy chơi game', 'máy fax', 'máy giặt', 'máy hút bụi', 'máy hút mùi',
                                                   'máy in', 'máy lạnh', 'máy mát xa', 'máy nghe nhạc',
                                                   'máy pha cafe', 'máy rửa bát', 'máy sưởi',
                                                   'máy tính', 'máy tính xách tay',
                                                   'quạt treo tường', 'quạt hút mùi', 'quạt thông gió',
                                                   'quạt trần', 'quạt', 'quạt hơi nước', 'quạt điện',
                                                   'quạt gió',
                                                   'tủ lạnh', 'tủ đá', 'tủ đông',
                                                   'van tưới',
                                                   'vòi tưới', 'vòi hoa sen', 'vòi sen', 'vòi tưới nước',
                                                   'vòi nước', 'vòi phun', 'vòi phun nước',
                                                   'điều hòa', 'máy lạnh', 'máy sưởi', 'máy điều hòa', 'máy làm mát',
                                                   'điều hòa nhiệt độ',
                                                   'điện',
                                                   'đèn bàn', 'đèn bếp', 'đèn chùm', 'đèn cây',
                                                   'đèn cổng', 'đèn học', 'đèn led', 'đèn làm việc',
                                                   'đèn ngủ', 'đèn sưởi', 'đèn thả', 'đèn tranh',
                                                   'đèn trụ cổng', 'đèn tuýp', 'đèn tường', 'đèn âm trần',
                                                   'đèn ốp tường', 'lò sưởi', 'lò nướng',
                                                   'lò vi sóng',
                                                   'máy nghe nhạc', 'máy phát nhạc', 'radio', 'máy chơi đĩa', 'máy CD',
                                                   'máy DVD',
                                                   'lò sưởi', 'lò nướng bánh mỳ', 'lò nướng bánh', 'lò vi sóng',
                                                   'lò vi ba',
                                                   'máy tính bảng', 'điện thoại thông minh', 'ipad', 'iphone'],
                                  'giảm mức độ của thiết bị': ['quạt hút mùi', 'quạt treo tường', 'quạt thông gió',
                                                               'quạt trần', 'quạt', 'quạt hơi nước', 'quạt điện',
                                                               'quạt gió', 'quạt đứng', 'quạt bàn', 'quạt máy',
                                                               'quạt hơi nước', 'quạt thông minh', 'quạt mini',
                                                               'quạt phun sương', 'quạt bếp', 'quạt tạo ẩm',
                                                               'phạt phun nước', 'quạt hút gió', 'quạt hút khí',
                                                               'quạt làm mát',
                                                               'vòi hoa sen', 'vòi sen', 'vòi tưới', 'vòi tưới nước',
                                                               'vòi nước', 'vòi tưới', 'vòi phun', 'vòi phun nước', ],
                                  'giảm nhiệt độ của thiết bị': ['máy lạnh', 'điều hòa', 'máy điều hòa', 'máy sưởi',
                                                                 'lò sười', 'đèn sưởi', 'đèn làm ấm',
                                                                 'điều hòa nhiệt độ', 'máy làm mát', 'lò vi sóng',
                                                                 'bình nóng lạnh', 'lò sưởi', 'lò nướng', 'bếp',
                                                                 'bếp điện',
                                                                 'bếp lửa', 'bếp từ', 'bếp ga', 'bộ điều hòa không khí',
                                                                 'nồi cơm điện', 'ấm đun nước', 'ấm điện',
                                                                 'ấm siêu tốc', 'máy pha cà phê',
                                                                 'bình đun nước', 'bình nhiệt', 'nồi hấp', 'nồi hơi',
                                                                 'nồi điện', 'nhiệt kế', 'nhiệt kế hồng ngoại',
                                                                 'lò vi ba', 'nồi hâm nóng', 'máy làm lạnh',
                                                                 'máy làm đá', 'máy pha cà phê', 'ấm đun nước điện tử',
                                                                 'máy làm bánh', 'lò nướng bánh'
                                                                 ],
                                  'giảm âm lượng của thiết bị': ["radio", "ti vi", "loa", "loa ti vi", "loa laptop",
                                                                 "loa máy tính", "loa thùng", "máy phát nhạc",
                                                                 "máy chơi nhạc", "cát xét", 'đầu đĩa', 'đầu chơi đĩa',
                                                                 'máy CD', 'máy DVD',
                                                                 'tai nghe', 'loa dàn', 'loa bluetooth', 'đầu đĩa CD',
                                                                 'đầu DVD', 'máy tính', 'máy tính bảng',
                                                                 'điện thoại thông minh', 'điện thoại', 'stereo',
                                                                 'máy ghi âm', 'dàn karraoke'],
                                  'giảm độ sáng của thiết bị': ['bóng', 'bóng chùm', 'bóng compact',
                                                                'bóng hắt', 'bóng làm việc', 'bóng ngủ',
                                                                'bóng sân', 'bóng thả', 'bóng treo tường',
                                                                'bóng tròn', 'bóng trụ cổng', 'bóng tuýp',
                                                                'bóng âm trần', 'bóng để bàn', 'bóng đứng',
                                                                'bóng ốp trần',
                                                                'bóng đèn', 'bóng đèn chùm', 'bóng đèn compact',
                                                                'bóng đèn hắt', 'bóng đèn làm việc', 'bóng đèn ngủ',
                                                                'bóng đèn sân', 'bóng đèn thả', 'bóng đèn treo tường',
                                                                'bóng đèn tròn', 'bóng đèn trụ cổng', 'bóng đèn tuýp',
                                                                'bóng đèn âm trần', 'bóng đèn để bàn', 'bóng đèn đứng',
                                                                'bóng đèn ốp trần',
                                                                'đèn', 'đèn bàn', 'đèn bếp',
                                                                'đèn chùm', 'đèn compact',
                                                                'đèn cây', 'đèn cổng', 'đèn huỳnh quang',
                                                                'đèn hắt', 'đèn hắt tường', 'đèn học',
                                                                'đèn led', 'đèn làm việc', 'đèn rủ',
                                                                'đèn sợi đốt', 'đèn thả',
                                                                'đèn tròn', 'đèn trụ cổng',
                                                                'đèn tuýp', 'đèn tường', 'đèn âm trần',
                                                                'đèn hẻm', 'đèn nền', 'đèn thông minh',
                                                                'đèn năng lượng mặt trời', 'màn hình máy tính',
                                                                'màn hình điện thoại',
                                                                'màn hình máy tính bảng', 'màn hình', 'đèn sân vườn',
                                                                'đèn đọc sách'],
                                  'hủy hoạt cảnh': [],
                                  'kiểm tra tình trạng thiết bị': ['bình nóng lạnh',
                                                                   'bóng chùm', 'bóng compact', 'bóng huỳnh quang',
                                                                   'bóng hắt',
                                                                   'bóng ngủ',
                                                                   'bóng đèn chùm', 'bóng đèn compact',
                                                                   'bóng đèn huỳnh quang',
                                                                   'bóng đèn hắt',
                                                                   'bóng đèn ngủ', 'bóng đèn sưởi',
                                                                   'bóng đèn treo tường',
                                                                   'bóng đèn tròn',
                                                                   'bóng đèn tuýp', 'bóng đèn đứng', 'bóng đèn ốp trần',
                                                                   'bếp', 'bếp điện', 'bếp ga', 'camera',
                                                                   'camera giám sát',
                                                                   'laptop', 'PC', 'loa',
                                                                   'loa ti vi', 'loa máy tính', 'lò nướng',
                                                                   'lò vi sóng',
                                                                   'lò sưởi',
                                                                   'máy chơi game', 'máy fax', 'máy giặt',
                                                                   'máy hút bụi', 'máy hút mùi',
                                                                   'máy in', 'máy lạnh', 'máy mát xa', 'máy nghe nhạc',
                                                                   'máy pha cafe', 'máy rửa bát', 'máy sưởi',
                                                                   'máy tính', 'máy tính xách tay',
                                                                   'quạt treo tường', 'quạt hút mùi', 'quạt thông gió',
                                                                   'quạt trần', 'quạt', 'quạt hơi nước', 'quạt điện',
                                                                   'quạt gió',
                                                                   'tủ lạnh', 'tủ đá', 'tủ đông',
                                                                   'van tưới',
                                                                   'vòi tưới', 'vòi hoa sen', 'vòi sen',
                                                                   'vòi tưới nước',
                                                                   'vòi nước', 'vòi phun', 'vòi phun nước',
                                                                   'điều hòa', 'máy lạnh', 'máy sưởi', 'máy điều hòa',
                                                                   'máy làm mát',
                                                                   'điều hòa nhiệt độ',
                                                                   'điện',
                                                                   'đèn bàn', 'đèn bếp', 'đèn chùm', 'đèn cây',
                                                                   'đèn cổng', 'đèn học', 'đèn led', 'đèn làm việc',
                                                                   'đèn ngủ', 'đèn sưởi', 'đèn thả', 'đèn tranh',
                                                                   'đèn trụ cổng', 'đèn tuýp', 'đèn tường',
                                                                   'đèn âm trần',
                                                                   'đèn ốp tường', 'đồng hồ', 'đồng hồ treo tường',
                                                                   'đồng hồ điện', 'điện thoại',
                                                                   'điện thoại thông minh',
                                                                   'máy tính bảng', 'iphone', 'ipad', 'máy phát nhạc',
                                                                   'máy chơi nhạc', 'đầu CD', 'đầu DVD'],
                                  'kích hoạt cảnh': [],
                                  'mở thiết bị': ['bình nóng lạnh',
                                                  'bóng chùm', 'bóng compact', 'bóng huỳnh quang', 'bóng hắt',
                                                  'bóng ngủ',
                                                  'bóng đèn chùm', 'bóng đèn compact', 'bóng đèn huỳnh quang',
                                                  'bóng đèn hắt',
                                                  'bóng đèn ngủ', 'bóng đèn sưởi', 'bóng đèn treo tường',
                                                  'bóng đèn tròn',
                                                  'bóng đèn tuýp', 'bóng đèn đứng', 'bóng đèn ốp trần',
                                                  'bếp', 'bếp điện', 'bếp ga', 'camera', 'camera giám sát'
                                                                                         'laptop', 'PC', 'loa',
                                                  'loa ti vi', 'loa máy tính', 'lò nướng', 'lò vi sóng',
                                                  'lò sưởi',
                                                  'máy chơi game', 'máy fax', 'máy giặt', 'máy hút bụi', 'máy hút mùi',
                                                  'máy in', 'máy lạnh', 'máy mát xa',
                                                  'máy pha cafe', 'máy rửa bát', 'máy sưởi',
                                                  'máy tính', 'máy tính xách tay',
                                                  'quạt treo tường', 'quạt hút mùi', 'quạt thông gió',
                                                  'quạt trần', 'quạt', 'quạt hơi nước', 'quạt điện',
                                                  'quạt gió',
                                                  'tủ lạnh', 'tủ đá', 'tủ đông', 'tủ đồ',
                                                  'van tưới',
                                                  'vòi tưới', 'vòi hoa sen', 'vòi sen', 'vòi tưới nước',
                                                  'vòi nước', 'vòi phun', 'vòi phun nước',
                                                  'điều hòa', 'máy lạnh', 'máy sưởi', 'máy điều hòa', 'máy làm mát',
                                                  'điều hòa nhiệt độ',
                                                  'điện',
                                                  'đèn bàn', 'đèn bếp', 'đèn chùm', 'đèn cây',
                                                  'đèn cổng', 'đèn học', 'đèn led', 'đèn làm việc',
                                                  'đèn ngủ', 'đèn sưởi', 'đèn thả', 'đèn tranh',
                                                  'đèn trụ cổng', 'đèn tuýp', 'đèn tường', 'đèn âm trần',
                                                  'đèn ốp tường', 'rèm cửa', 'rèm cửa sổ', 'cửa cuốn', 'cửa sổ',
                                                  'cửa cổng', 'cổng', 'cửa ra vào', 'cửa', 'cổng ra vào', 'rèm ngủ',
                                                  'laptop', 'máy tính bảng', 'cửa tủ', 'rèm', 'màn ngủ', 'màn cuốn',
                                                  'màn cửa', 'radio', 'stereo', 'rèm cửa', 'rèm cửa sổ',
                                                  'mành cửa', 'mành cuốn', 'camera', 'lò nướng',
                                                  'lò vi sóng',
                                                  'lò sưởi', ],
                                  'tăng mức độ của thiết bị': ['quạt hút mùi', 'quạt treo tường', 'quạt thông gió',
                                                               'quạt trần', 'quạt', 'quạt hơi nước', 'quạt điện',
                                                               'quạt gió', 'quạt đứng', 'quạt bàn', 'quạt máy',
                                                               'quạt hơi nước', 'quạt thông minh', 'quạt mini',
                                                               'quạt phun sương', 'quạt bếp', 'quạt tạo ẩm',
                                                               'phạt phun nước', 'quạt hút gió', 'quạt hút khí',
                                                               'quạt làm mát',
                                                               'vòi hoa sen', 'vòi sen', 'vòi tưới', 'vòi tưới nước',
                                                               'vòi nước', 'vòi tưới', 'vòi phun', 'vòi phun nước', ],
                                  'tăng nhiệt độ của thiết bị': ['máy lạnh', 'điều hòa', 'máy điều hòa', 'máy sưởi',
                                                                 'lò sười', 'đèn sưởi', 'đèn làm ấm',
                                                                 'điều hòa nhiệt độ', 'máy làm mát', 'lò vi sóng',
                                                                 'bình nóng lạnh', 'lò sưởi', 'lò nướng', 'bếp',
                                                                 'bếp điện',
                                                                 'bếp lửa', 'bếp từ', 'bếp ga', 'bộ điều hòa không khí',
                                                                 'nồi cơm điện', 'ấm đun nước', 'ấm điện',
                                                                 'ấm siêu tốc', 'máy pha cà phê',
                                                                 'bình đun nước', 'bình nhiệt', 'nồi hấp', 'nồi hơi',
                                                                 'nồi điện', 'nhiệt kế', 'nhiệt kế hồng ngoại',
                                                                 'lò vi ba', 'nồi hâm nóng', 'máy làm lạnh',
                                                                 'máy làm đá', 'máy pha cà phê', 'ấm đun nước điện tử',
                                                                 'máy làm bánh', 'lò nướng bánh'
                                                                 ],
                                  'tăng âm lượng của thiết bị': ["radio", "ti vi", "loa", "loa ti vi", "loa laptop",
                                                                 "loa máy tính", "loa thùng", "máy phát nhạc",
                                                                 "máy chơi nhạc", "cát xét", 'đầu đĩa', 'đầu chơi đĩa',
                                                                 'máy CD', 'máy DVD',
                                                                 'tai nghe', 'loa dàn', 'loa bluetooth', 'đầu đĩa CD',
                                                                 'đầu DVD', 'máy tính', 'máy tính bảng',
                                                                 'điện thoại thông minh', 'điện thoại', 'stereo',
                                                                 'máy ghi âm', 'dàn karraoke'],

                                  'tăng độ sáng của thiết bị': ['bóng', 'bóng chùm', 'bóng compact',
                                                                'bóng hắt', 'bóng làm việc', 'bóng ngủ',
                                                                'bóng sân', 'bóng thả', 'bóng treo tường',
                                                                'bóng tròn', 'bóng trụ cổng', 'bóng tuýp',
                                                                'bóng âm trần', 'bóng để bàn', 'bóng đứng',
                                                                'bóng ốp trần',
                                                                'bóng đèn', 'bóng đèn chùm', 'bóng đèn compact',
                                                                'bóng đèn hắt', 'bóng đèn làm việc', 'bóng đèn ngủ',
                                                                'bóng đèn sân', 'bóng đèn thả', 'bóng đèn treo tường',
                                                                'bóng đèn tròn', 'bóng đèn trụ cổng', 'bóng đèn tuýp',
                                                                'bóng đèn âm trần', 'bóng đèn để bàn', 'bóng đèn đứng',
                                                                'bóng đèn ốp trần',
                                                                'đèn', 'đèn bàn', 'đèn bếp',
                                                                'đèn chùm', 'đèn compact',
                                                                'đèn cây', 'đèn cổng', 'đèn huỳnh quang',
                                                                'đèn hắt', 'đèn hắt tường', 'đèn học',
                                                                'đèn led', 'đèn làm việc', 'đèn rủ',
                                                                'đèn sợi đốt', 'đèn thả',
                                                                'đèn tròn', 'đèn trụ cổng',
                                                                'đèn tuýp', 'đèn tường', 'đèn âm trần',
                                                                'đèn hẻm', 'đèn nền', 'đèn thông minh',
                                                                'đèn năng lượng mặt trời', 'màn hình máy tính',
                                                                'màn hình điện thoại',
                                                                'màn hình máy tính bảng', 'màn hình', 'đèn sân vườn',
                                                                'đèn đọc sách'],
                                  'tắt thiết bị': ['bình nóng lạnh',
                                                   'bóng chùm', 'bóng compact', 'bóng huỳnh quang', 'bóng hắt',
                                                   'bóng ngủ',
                                                   'bóng đèn chùm', 'bóng đèn compact', 'bóng đèn huỳnh quang',
                                                   'bóng đèn hắt',
                                                   'bóng đèn ngủ', 'bóng đèn sưởi', 'bóng đèn treo tường',
                                                   'bóng đèn tròn',
                                                   'bóng đèn tuýp', 'bóng đèn đứng', 'bóng đèn ốp trần',
                                                   'bếp', 'bếp điện', 'bếp ga', 'camera', 'camera giám sát',
                                                   'laptop', 'PC', 'loa',
                                                   'loa ti vi', 'loa máy tính', 'lò nướng', 'lò vi sóng',
                                                   'lò sưởi',
                                                   'máy chơi game', 'máy fax', 'máy giặt', 'máy hút bụi', 'máy hút mùi',
                                                   'máy in', 'máy lạnh', 'máy mát xa', 'máy nghe nhạc',
                                                   'máy pha cafe', 'máy rửa bát', 'máy sưởi',
                                                   'máy tính', 'máy tính xách tay',
                                                   'quạt treo tường', 'quạt hút mùi', 'quạt thông gió',
                                                   'quạt trần', 'quạt', 'quạt hơi nước', 'quạt điện',
                                                   'quạt gió',
                                                   'tủ lạnh', 'tủ đá', 'tủ đông',
                                                   'van tưới',
                                                   'vòi tưới', 'vòi hoa sen', 'vòi sen', 'vòi tưới nước',
                                                   'vòi nước', 'vòi phun', 'vòi phun nước',
                                                   'điều hòa', 'máy lạnh', 'máy sưởi', 'máy điều hòa', 'máy làm mát',
                                                   'điều hòa nhiệt độ',
                                                   'điện',
                                                   'đèn bàn', 'đèn bếp', 'đèn chùm', 'đèn cây',
                                                   'đèn cổng', 'đèn học', 'đèn led', 'đèn làm việc',
                                                   'đèn ngủ', 'đèn sưởi', 'đèn thả', 'đèn tranh',
                                                   'đèn trụ cổng', 'đèn tuýp', 'đèn tường', 'đèn âm trần',
                                                   'đèn ốp tường', 'máy nghe nhạc', 'máy phát nhạc', 'radio',
                                                   'máy chơi đĩa', 'máy CD',
                                                   'máy DVD', ],
                                  'đóng thiết bị': ['bình nóng lạnh',
                                                    'bóng chùm', 'bóng compact', 'bóng huỳnh quang', 'bóng hắt',
                                                    'bóng ngủ',
                                                    'bóng đèn chùm', 'bóng đèn compact', 'bóng đèn huỳnh quang',
                                                    'bóng đèn hắt',
                                                    'bóng đèn ngủ', 'bóng đèn sưởi', 'bóng đèn treo tường',
                                                    'bóng đèn tròn',
                                                    'bóng đèn tuýp', 'bóng đèn đứng', 'bóng đèn ốp trần',
                                                    'bếp', 'bếp điện', 'bếp ga', 'camera', 'camera giám sát'
                                                                                           'laptop', 'PC', 'loa',
                                                    'loa ti vi', 'loa máy tính', 'lò nướng', 'lò vi sóng',
                                                    'lò sưởi',
                                                    'máy chơi game', 'máy fax', 'máy giặt', 'máy hút bụi',
                                                    'máy hút mùi',
                                                    'máy in', 'máy lạnh', 'máy mát xa', 'máy nghe nhạc',
                                                    'máy pha cafe', 'máy rửa bát', 'máy sưởi',
                                                    'máy tính', 'máy tính xách tay',
                                                    'quạt treo tường', 'quạt hút mùi', 'quạt thông gió',
                                                    'quạt trần', 'quạt', 'quạt hơi nước', 'quạt điện',
                                                    'quạt gió',
                                                    'tủ lạnh', 'tủ đá', 'tủ đông', 'tủ đồ',
                                                    'van tưới',
                                                    'vòi tưới', 'vòi hoa sen', 'vòi sen', 'vòi tưới nước',
                                                    'vòi nước', 'vòi phun', 'vòi phun nước',
                                                    'điều hòa', 'máy lạnh', 'máy sưởi', 'máy điều hòa', 'máy làm mát',
                                                    'điều hòa nhiệt độ',
                                                    'điện',
                                                    'đèn bàn', 'đèn bếp', 'đèn chùm', 'đèn cây',
                                                    'đèn cổng', 'đèn học', 'đèn led', 'đèn làm việc',
                                                    'đèn ngủ', 'đèn sưởi', 'đèn thả', 'đèn tranh',
                                                    'đèn trụ cổng', 'đèn tuýp', 'đèn tường', 'đèn âm trần',
                                                    'đèn ốp tường', 'rèm cửa', 'rèm cửa sổ', 'cửa cuốn', 'cửa sổ',
                                                    'cửa cổng', 'cổng', 'cửa ra vào', 'cửa', 'cổng ra vào', 'rèm ngủ',
                                                    'laptop', 'máy tính bảng', 'cửa tủ', 'rèm', 'màn ngủ', 'màn cuốn',
                                                    'màn cửa', 'radio', 'stereo', 'rèm cửa', 'rèm cửa sổ',
                                                    'mành cửa', 'mành cuốn', 'camera']}

possible_intent_command_mapping = {'bật thiết bị': ['kích hoạt', "khởi động", 'mở', 'chạy'],
                                   'giảm mức độ của thiết bị': ['hạ', 'cho thấp', 'cho nhỏ', 'làm nhỏ', 'cho bé',
                                                                'làm bé', 'vặn nhỏ', 'đưa xuống'],
                                   'giảm nhiệt độ của thiết bị': ['hạ', 'hạ nhiệt', 'giảm nhiệt', 'làm mát', 'làm lạnh',
                                                                  'cho thấp', 'cho bé', 'cho nhỏ', 'chỉnh thấp',
                                                                  'chỉnh nhỏ', 'chỉnh bé', 'bật bé', 'đưa lên'],
                                   'giảm âm lượng của thiết bị': ['hạ', 'cho nhỏ', 'hạ bé', 'cho bé', 'bớt', 'bật nhỏ',
                                                                  'bật bé'],
                                   'giảm độ sáng của thiết bị': ['hạ', 'làm tối', 'cho tối', 'giảm sáng', 'bớt',
                                                                 'làm mờ', 'cho mờ', 'bật bé', 'bật nhỏ'],
                                   'hủy hoạt cảnh': [],
                                   'kiểm tra tình trạng thiết bị': ['check', "chếch", 'xem', 'xem xét', 'xem lại'],
                                   'kích hoạt cảnh': [],
                                   'mở thiết bị': ['mở'],
                                   'tăng mức độ của thiết bị': ['nâng', 'làm to', 'cho to', 'vặn to', 'vặn lớn',
                                                                'cho lớn', 'bật to', 'đưa lên'],
                                   'tăng nhiệt độ của thiết bị': ['nâng', 'làm ấm', 'làm nóng', 'cho to', 'cho lớn',
                                                                  'bật to', 'đưa lên'],
                                   'tăng âm lượng của thiết bị': ['nâng', 'vặn to', 'cho to', 'bật to', 'bật to',
                                                                  'bật lớn', 'đưa lên'],
                                   'tăng độ sáng của thiết bị': ['nâng', 'làm sáng', 'cho sáng', 'bật sáng'],
                                   'tắt thiết bị': ["ngừng", "dừng", 'ngắt', 'sập nguồn', 'ngắt nguồn', 'ngắt điện'],
                                   'đóng thiết bị': ['khóa', 'sập', 'chốt']}
bad_command = ['cho thấp', 'cho nhỏ', 'làm nhỏ', 'cho bé',
               'làm bé', 'bật bé', 'cho nhỏ', 'làm mát', 'làm lạnh',
               'cho thấp', 'cho bé', 'cho nhỏ', 'cho nhỏ', 'hạ bé', 'cho bé', 'bật nhỏ',
               'bật bé', 'làm tối', 'cho tối', 'giảm sáng', 'bớt',
               'làm mờ', 'cho mờ', 'bật bé', 'bật nhỏ',
               'làm to', 'cho to', 'làm ấm', 'làm nóng', 'cho to', 'cho lớn',
               'cho lớn', 'bật to', 'bật lớn', 'làm sáng', 'cho sáng', 'bật sáng',
               'đưa lên', 'đưa lên'
               ]
special_command = ['vặn', 'bật']

need_to_change_prefix = {
    "changing value": ["anh còn", "xuống còn", "còn khoảng", "còn mức", "còn số", "còn tầm", "về", "còn", "đến",
                       "về mức", "lên mức", "xuống mức", "ở mức", "mức số", "mức độ", "đến mức",
                       "trước sang", "lên số", "xuống số",
                       "chùm thành", "trần thành", "đốt thành", "xuống thành",
                       "4 đến", "radio đến", "sáng đến", "tăng đến", "bật đến", "cổng đến", "giảm đến", "học đến",
                       "lang đến", "lên đến", "quân đến", "sưởi đến", "tuýp đến", "tường đến", "đèn đến", "đến khoảng",
                       "đến mức",
                       "tăng đến", "giảm đến", "cổng về", "tuýp về"],

}

text_num_mapping = {'ba': 3,
                    'ba ba': 33,
                    'ba bảy': 37,
                    'ba bốn': 34,
                    'ba chín': 39,
                    'ba hai': 32,
                    'ba lăm': 35,
                    'ba mươi': 30,
                    'ba mươi ba': 33,
                    'ba mươi bảy': 37,
                    'ba mươi bốn': 34,
                    'ba mươi chín': 39,
                    'ba mươi hai': 32,
                    'ba mươi lăm': 35,
                    'ba mươi mốt': 31,
                    'ba mươi năm': 35,
                    'ba mươi sáu': 36,
                    'ba mươi tám': 38,
                    'ba mươi tư': 34,
                    'ba mốt': 31,
                    'ba năm': 35,
                    'ba sáu': 36,
                    'ba tám': 38,
                    'ba tư': 34,
                    'bảy': 7,
                    'bảy ba': 73,
                    'bảy bảy': 77,
                    'bảy bốn': 74,
                    'bảy chín': 79,
                    'bảy hai': 72,
                    'bảy lăm': 75,
                    'bảy mươi': 70,
                    'bảy mươi ba': 73,
                    'bảy mươi bảy': 77,
                    'bảy mươi bốn': 74,
                    'bảy mươi chín': 79,
                    'bảy mươi hai': 72,
                    'bảy mươi lăm': 75,
                    'bảy mươi mốt': 71,
                    'bảy mươi năm': 75,
                    'bảy mươi sáu': 76,
                    'bảy mươi tám': 78,
                    'bảy mươi tư': 74,
                    'bảy mốt': 71,
                    'bảy năm': 75,
                    'bảy sáu': 76,
                    'bảy tám': 78,
                    'bảy tư': 74,
                    'bốn': 4,
                    'bốn ba': 43,
                    'bốn bảy': 47,
                    'bốn bốn': 44,
                    'bốn chín': 49,
                    'bốn hai': 42,
                    'bốn lăm': 45,
                    'bốn mươi': 40,
                    'bốn mươi ba': 43,
                    'bốn mươi bảy': 47,
                    'bốn mươi bốn': 44,
                    'bốn mươi chín': 49,
                    'bốn mươi hai': 42,
                    'bốn mươi lăm': 45,
                    'bốn mươi mốt': 41,
                    'bốn mươi năm': 45,
                    'bốn mươi sáu': 46,
                    'bốn mươi tám': 48,
                    'bốn mươi tư': 44,
                    'bốn mốt': 41,
                    'bốn năm': 45,
                    'bốn sáu': 46,
                    'bốn tám': 48,
                    'bốn tư': 44,
                    'chín': 9,
                    'chín ba': 93,
                    'chín bảy': 97,
                    'chín bốn': 94,
                    'chín chín': 99,
                    'chín hai': 92,
                    'chín lăm': 95,
                    'chín mươi': 90,
                    'chín mươi ba': 93,
                    'chín mươi bảy': 97,
                    'chín mươi bốn': 94,
                    'chín mươi chín': 99,
                    'chín mươi hai': 92,
                    'chín mươi lăm': 95,
                    'chín mươi mốt': 91,
                    'chín mươi năm': 95,
                    'chín mươi sáu': 96,
                    'chín mươi tám': 98,
                    'chín mươi tư': 94,
                    'chín mốt': 91,
                    'chín năm': 95,
                    'chín sáu': 96,
                    'chín tám': 98,
                    'chín tư': 94,
                    'hai': 2,
                    'hai ba': 23,
                    'hai bảy': 27,
                    'hai bốn': 24,
                    'hai chín': 29,
                    'hai hai': 22,
                    'hai lăm': 25,
                    'hai mươi': 20,
                    'hai mươi ba': 23,
                    'hai mươi bảy': 27,
                    'hai mươi bốn': 24,
                    'hai mươi chín': 29,
                    'hai mươi hai': 22,
                    'hai mươi lăm': 25,
                    'hai mươi mốt': 21,
                    'hai mươi năm': 25,
                    'hai mươi sáu': 26,
                    'hai mươi tám': 28,
                    'hai mươi tư': 24,
                    'hai mốt': 21,
                    'hai năm': 25,
                    'hai sáu': 26,
                    'hai tám': 28,
                    'hai tư': 24,
                    'lăm': 5,
                    'lăm ba': 53,
                    'lăm bảy': 57,
                    'lăm bốn': 54,
                    'lăm chín': 59,
                    'lăm hai': 52,
                    'lăm lăm': 55,
                    'lăm mươi': 50,
                    'lăm mươi ba': 53,
                    'lăm mươi bảy': 57,
                    'lăm mươi bốn': 54,
                    'lăm mươi chín': 59,
                    'lăm mươi hai': 52,
                    'lăm mươi lăm': 55,
                    'lăm mươi mốt': 51,
                    'lăm mươi năm': 55,
                    'lăm mươi sáu': 56,
                    'lăm mươi tám': 58,
                    'lăm mươi tư': 54,
                    'lăm mốt': 51,
                    'lăm năm': 55,
                    'lăm sáu': 56,
                    'lăm tám': 58,
                    'lăm tư': 54,
                    'mười': 10,
                    'mười ba': 13,
                    'mười bảy': 17,
                    'mười bốn': 14,
                    'mười chín': 19,
                    'mười hai': 12,
                    'mười lăm': 15,
                    'mười một': 11,
                    'mười năm': 15,
                    'mười sáu': 16,
                    'mười tám': 18,
                    'một': 1,
                    'năm': 5,
                    'năm ba': 53,
                    'năm bảy': 57,
                    'năm bốn': 54,
                    'năm chín': 59,
                    'năm hai': 52,
                    'năm lăm': 55,
                    'năm mươi': 50,
                    'năm mươi ba': 53,
                    'năm mươi bảy': 57,
                    'năm mươi bốn': 54,
                    'năm mươi chín': 59,
                    'năm mươi hai': 52,
                    'năm mươi lăm': 55,
                    'năm mươi mốt': 51,
                    'năm mươi năm': 55,
                    'năm mươi sáu': 56,
                    'năm mươi tám': 58,
                    'năm mươi tư': 54,
                    'năm mốt': 51,
                    'năm năm': 55,
                    'năm sáu': 56,
                    'năm tám': 58,
                    'năm tư': 54,
                    'sáu': 6,
                    'sáu ba': 63,
                    'sáu bảy': 67,
                    'sáu bốn': 64,
                    'sáu chín': 69,
                    'sáu hai': 62,
                    'sáu lăm': 65,
                    'sáu mươi': 60,
                    'sáu mươi ba': 63,
                    'sáu mươi bảy': 67,
                    'sáu mươi bốn': 64,
                    'sáu mươi chín': 69,
                    'sáu mươi hai': 62,
                    'sáu mươi lăm': 65,
                    'sáu mươi mốt': 61,
                    'sáu mươi năm': 65,
                    'sáu mươi sáu': 66,
                    'sáu mươi tám': 68,
                    'sáu mươi tư': 64,
                    'sáu mốt': 61,
                    'sáu năm': 65,
                    'sáu sáu': 66,
                    'sáu tám': 68,
                    'sáu tư': 64,
                    'tám': 8,
                    'tám ba': 83,
                    'tám bảy': 87,
                    'tám bốn': 84,
                    'tám chín': 89,
                    'tám hai': 82,
                    'tám lăm': 85,
                    'tám mươi': 80,
                    'tám mươi ba': 83,
                    'tám mươi bảy': 87,
                    'tám mươi bốn': 84,
                    'tám mươi chín': 89,
                    'tám mươi hai': 82,
                    'tám mươi lăm': 85,
                    'tám mươi mốt': 81,
                    'tám mươi năm': 85,
                    'tám mươi sáu': 86,
                    'tám mươi tám': 88,
                    'tám mươi tư': 84,
                    'tám mốt': 81,
                    'tám năm': 85,
                    'tám sáu': 86,
                    'tám tám': 88,
                    'tám tư': 84}
# device, command, target number, time at, location, changing value, duration, scene

possible_confusing_device = ['máy tính', 'nhà thông minh', 'trợ lý ảo', 'điện thoại thông minh', 'điện thoại', 'trợ lý',
                             'AI', 'trí tuệ nhân tạo', 'robot', 'người máy']
possible_verbal_words = ['ơi', 'à', 'ây', 'êi', 'này', 'nè']

possible_confusion_words = ["à", "à mà thôi", "à thôi", "à nhầm", "nhầm", "à đâu", "à quên"]

scenes = [
    'thư giãn',
    'lãng mạn',
    'ra khỏi phòng',
    'riêng tư',
    'thư giãn',
    'tiệc tùng',
    'đi ngủ', 'máy CD', 'máy DVD',

    'đi tắm',
    'đi chơi',
    'yên tĩnh',
    'yên lặng',
    'sôi động',
    'ra ngoài',
    'giải trí',
    'nghiêm túc',
    'im lặng',
    "ở ngoài trời",
    "ở trong phòng"
]

subject_list = [
    "tôi",
    "mình",
    "tớ",
    "tui",
    "anh",
    'chị',
    'em',
    "tao",
    "anh ấy",
    "em ấy",
    "chị ấy",
    "bạn ấy",
]

support_words = [
    "sẽ",
    "đang",
    "vừa",
    "muốn",
    "định",
    "không định",
    "chẳng muốn",
    "không muốn",
    "thích",
    "chẳng thích",
    "không thích",
    "cần",
    "không cần",
    "chẳng cần",
    "vẫn đang",
    "còn đang",
    "đang cần",
    "vừa mới",
    "mới"
]

end_words = [
    "đấy",
    "nhỉ",
    "mà nhỉ",
    "đấy nhỉ",
    "mà",
    "nữa",
    "",
]


def generate_sentence():
    template = "{} {} {} {}"
    first_slot = random.choice(subject_list)
    second_slot = random.choice(support_words)
    third_slot = random.choice(scenes)
    fourth_slot = random.choice(end_words)
    return template.format(first_slot, second_slot, third_slot, fourth_slot)


intent_rate_mapping = {

}

list_prefix_verbs = possible_intent_command_mapping['kiểm tra tình trạng thiết bị']
list_subject = ['em', 'anh', 'chị', 'tớ', 'tôi', 'tao', 'chúng tôi', 'ta', 'chúng ta', 'bọn tôi', 'bọn anh', 'bọn chị',
                'anh ấy', 'chị ấy']  # TODO: thêm tên riêng
list_linking_verb_default = ['cần biết', 'muốn biết', 'muốn hỏi', 'cần biết', 'cần xem', 'cần hỏi']
list_linking_verb_include_command = ['cần ' + c for c in list_prefix_verbs] + ['muốn ' + c for c in list_prefix_verbs]


def create_prefix(include_postfix=False):
    choice = random.random()
    subject = None
    prefix_annotation = ''

    if choice < 0.3:  # a muốn xem
        type = 1
        subject = random.choice(list_subject)
        command_in_linking_verb = random.random() < 0.3
        if command_in_linking_verb:
            linking_verb = random.choice(list_linking_verb_default)
        else:
            linking_verb = random.choice(list_linking_verb_include_command)
            linking_verb_command = " ".join(linking_verb.split()[1:])

        prefix_head = random.choice(['đang', 'vẫn', 'rất'])
        prefix = "{} {} {}".format(subject, prefix_head,
                                   linking_verb)
        if command_in_linking_verb:
            prefix_annotation = prefix
        else:
            prefix_annotation = "{} {} {}".format(subject, prefix_head,
                                                  linking_verb.split()[0] + f" [ command : {linking_verb_command} ]")
    elif choice < 0.5:  # xem cho a
        type = 2
        subject = random.choice(list_subject)

        command = random.choice(list_prefix_verbs)
        # subject = random.choice(['cho ' + subject, ''])
        prefix_tail = random.choice(['cho ' + subject, ''])
        prefix = '{} {}'.format(command,
                                prefix_tail)
        if include_postfix:
            prefix_annotation = prefix
        else:
            prefix_annotation = f'[ command : {command} ] {prefix_tail}'
    else:
        type = 3
        prefix = ''
    while "  " in prefix:
        prefix = prefix.replace("  ", ' ')
    while "  " in prefix_annotation:
        prefix_annotation.replace("  ", " ")
    return prefix.strip(), type, subject, prefix_annotation.strip()


state_prefix = ['còn', 'có', 'đang', 'có đang', 'vẫn đang', 'vẫn còn', 'có còn']
intent_state_mapping = {
    'bật thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn', 'chạy tốt'],
    'giảm mức độ của thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn',
                                 'chạy tốt'],
    'giảm nhiệt độ của thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn',
                                   'chạy tốt', 'nóng',
                                   'ấm', ],
    'giảm âm lượng của thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn',
                                   'chạy tốt', 'to',
                                   'lớn', 'ồn'],
    'giảm độ sáng của thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn',
                                  'chạy tốt', 'sáng',
                                  'chói'],
    'hủy hoạt cảnh': [],
    'kiểm tra tình trạng thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn',
                                     'chạy tốt'],
    'kích hoạt cảnh': [],
    'mở thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn', 'chạy tốt'],
    'tăng mức độ của thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn',
                                 'chạy tốt', 'yếu', 'chạy yếu'],
    'tăng nhiệt độ của thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn',
                                   'chạy tốt', 'lạnh',
                                   'mát'],
    'tăng âm lượng của thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn',
                                   'chạy tốt',
                                   'nhỏ', 'bé'],
    'tăng độ sáng của thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn',
                                  'chạy tốt',
                                  'tối', 'mờ', 'thiếu sáng'],
    'tắt thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn', 'chạy tốt'],
    'đóng thiết bị': ['mở', 'chạy', 'hoạt động', 'dùng được', 'dùng tốt', 'dùng ổn', 'sống', 'chạy ổn', 'chạy tốt']}

middle_postfix_commad = ['không vậy ?', 'không nhỉ ?', 'hay không ?', 'à ?', 'không vậy .', 'không nhỉ .',
                         'hay không .', 'à .', 'không vậy ,', 'không nhỉ ,', 'hay không ,', 'à ,']
middle_postfix_no_commad = ['nhé', 'nha', 'nhá', 'không', 'nhở']


def create_middle(intent, include_postfix=False):
    middle_device = possible_intent_device_mapping[intent]
    state = intent_state_mapping[intent]
    choice = random.random()
    label = ''
    middle_prefix = ''
    if choice < 0.3:
        location_prefix = random.choice(['ở ', 'trong ', 'ngoài ', 'gần ', 'bên ', 'cạnh ', 'trên ',
                                         'dưới '])
        location = random.choice(location_list)
        one, two, three, four, five, six = (random.choice(['cái', 'chiếc', 'con', 'cái con', 'thằng', 'cái thằng', '']),
                                            random.choice(middle_device),
                                            random.choice([location_prefix + location, '']),
                                            random.choice(state_prefix),
                                            random.choice(state),
                                            random.choice(
                                                middle_postfix_commad if include_postfix else middle_postfix_no_commad))

        middle_prefix = random.choice(['hình như ', 'hay là ', ''])
        middle = "{} {} {} {} {} {}".format(one, two, three, four, five, six)
        middle = middle_prefix + middle
        label = middle_prefix + f"{one} [ device : {two} ] " + (
            '' if three == '' else f'{location_prefix}  [ location : {location} ] ') + f'{four} {five} {six}'
    elif choice < 0.6:
        location_prefix = random.choice(['ở ', 'trong ', 'ngoài ', 'gần ', 'bên ', 'cạnh ', 'trên ',
                                         'dưới '])
        location = random.choice(location_list)
        one, two, three, four, five, six = (random.choice([location_prefix + location, '']),
                                            random.choice(['cái', 'chiếc', 'con', 'cái con', 'thằng', 'cái thằng', '']),
                                            random.choice(middle_device),

                                            random.choice(state_prefix),
                                            random.choice(state),
                                            random.choice(
                                                middle_postfix_commad if include_postfix else middle_postfix_no_commad))
        middle_prefix = random.choice(['hình như ', 'hay là ', ''])

        middle = "{} {} {} {} {} {}".format(one, two, three, four, five, six)
        middle = middle_prefix + middle
        label = middle_prefix + (
            '' if one == '' else f'{location_prefix}  [ location : {location} ] ') + f"{two} [ device : {three} ] {four} {five} {six}"

    else:
        location_prefix = random.choice(['ở ', 'trong ', 'ngoài ', 'gần ', 'bên ', 'cạnh ', 'trên ',
                                         'dưới '])
        location = random.choice(location_list)
        one, two, three, four, five, six = (random.choice(['cái', 'chiếc', 'con', 'cái con', 'thằng', 'cái thằng', '']),
                                            random.choice(middle_device),
                                            random.choice(state_prefix),
                                            random.choice(state),
                                            random.choice([location_prefix + location, '']),
                                            random.choice(
                                                middle_postfix_commad if include_postfix else middle_postfix_no_commad))

        middle = "{} {} {} {} {} {}".format(one, two, three, four, five, six)
        middle = middle_prefix + middle
        label = middle_prefix + f"{one} [ device : {two} ] " + f'{three} {four} ' + (
            '' if five == '' else f'{location_prefix}  [ location : {location} ] ') + f"{six}"
    while "  " in middle:
        middle = middle.replace("  ", " ")
    while "  " in label:
        label = label.replace("  ", " ")
    return middle.strip(), label.strip()


subject_postfix = ['hộ', 'cho', 'giúp', 'dùm']


def create_postfix(intent, subject, type=1):
    label = ''
    if subject is None:
        subject = random.choice(subject_list)
    if type != 2 and random.random() < 0.7:
        postfix_command = possible_intent_command_mapping[intent]
        command = random.choice(postfix_command)
        one, two, three = (command,
                           random.choice([random.choice(subject_postfix) + ' ' + subject, '']),
                           random.choice(['nhé', 'nhá', 'nhớ', 'được không', 'đi']))
        postfix = "{} {} {}".format(one, two, three)
        label = f"[ command : {one} ] {two} {three}"
    else:
        postfix = ''
        type = 2
    while "  " in postfix:
        postfix = postfix.replace("  ", " ")
    while "  " in label:
        label = label.replace("  ", " ")
    return postfix.strip(), label.strip(), type



opposite_intent_mapping = {'bật thiết bị': 'tắt thiết bị',
                           'giảm mức độ của thiết bị': 'tăng mức độ của thiết bị',
                           'giảm nhiệt độ của thiết bị': 'tăng nhiệt độ của thiết bị',
                           'giảm âm lượng của thiết bị': 'tăng âm lượng của thiết bị',
                           'giảm độ sáng của thiết bị': 'tăng độ sáng của thiết bị',
                           'hủy hoạt cảnh': 'kích hoạt cảnh',
                           'kiểm tra tình trạng thiết bị': None,
                           'kích hoạt cảnh': 'hủy hoạt cảnh',
                           'mở thiết bị': 'đóng thiết bị',
                           'tăng mức độ của thiết bị': 'giảm mức độ của thiết bị',
                           'tăng nhiệt độ của thiết bị': 'giảm nhiệt độ của thiết bị',
                           'tăng âm lượng của thiết bị': 'giảm âm lượng của thiết bị',
                           'tăng độ sáng của thiết bị': 'giảm độ sáng của thiết bị',
                           'tắt thiết bị': 'bật thiết bị',
                           'đóng thiết bị': 'mở thiết bị'}
neutral_intent = ['bật thiết bị', 'tắt thiết bị', 'mở thiết bị', 'đóng thiết bị']
time_repeat_freq = [
    "mỗi ngày",
    "từng ngày",
    "hàng ngày",
    "hằng ngày",
    "ngày mai",
    "ngày kia",
    # "hôm qua",
    # "hôm kia",
    # "hôm nọ"
]
reversed_intent_mapping = {
    'bật thiết bị': 'tắt thiết bị',
    'tắt thiết bị': 'bật thiết bị',
    'giảm mức độ của thiết bị': 'tăng mức độ của thiết bị',
    'tăng mức độ của thiết bị': 'giảm mức độ của thiết bị',
    'giảm nhiệt độ của thiết bị': 'tăng nhiệt độ của thiết bị',
    'tăng nhiệt độ của thiết bị': 'giảm nhiệt độ của thiết bị',
    'giảm âm lượng của thiết bị': 'tăng âm lượng của thiết bị',
    'tăng âm lượng của thiết bị': 'giảm âm lượng của thiết bị',
    'giảm độ sáng của thiết bị': 'tăng độ sáng của thiết bị',
    'tăng độ sáng của thiết bị': 'giảm độ sáng của thiết bị',
    'hủy hoạt cảnh': 'kích hoạt cảnh',
    'kích hoạt cảnh': 'hủy hoạt cảnh',
    'mở thiết bị': 'đóng thiết bị',
    'đóng thiết bị': 'mở thiết bị'
}

reversed_command_prefix = [
    "đừng",
    "không được",
    "đừng có",
    "đừng có mà",
    "làm ơn đừng",
    "chớ",
    "nghiêm cấm",
    "cấm"
]
device_keyword_command_mapping_dong_thiet_bi = {
    'cửa': ['xếp', 'gấp', 'khép', 'sập', 'kéo'],
    'rèm': ['xếp', 'gấp', 'khép', 'kéo'],
    'vòi': ['vặn'],
    'màn': ['xếp', 'gấp', 'kéo']
}
device_keyword_command_mapping_mo_thiet_bi = {
    'cửa': ['kéo'],
    'rèm': ['kéo'],
    'màn': ['kéo']

}
device_keyword_command_mapping_bat_thiet_bi = {
    'radio': ['phát'],
    'CD': ['phát'],
    'DVD': ['phát'],
    'nhạc': ['phát'],
    'lò': ['quay'],

}

human_names = [
    "Ái", "An", "Bắc", "Bạc", "Băng", "Bảo", "Bích", "Binh", "Cẩn", "Cát",
    "Chân", "Châu", "Chi", "Chiến", "Chinh", "Cung", "Cường", "Đài", "Đăng",
    "Đào", "Đất", "Đe", "Di", "Diễm", "Diệp", "Dĩnh", "Đoan", "Đức", "Dung",
    "Dương", "Duy", "Duyên", "Giang", "Hà", "Hạ", "Hải", "Hân", "Hằng", "Hạnh",
    "Hiền", "Hiếu", "Hoàng", "Huân", "Hùng", "Hương", "Hưng", "Huy", "Huyền",
    "Huỳnh", "Khắc", "Khánh", "Khiêm", "Khoa", "Khuê", "Kiệt", "Kỳ", "Lá",
    "Lam", "Lâm", "Lan", "Lễ", "Lệ", "Liên", "Linh", "Lộc", "Long", "Lực",
    "Lý", "Mai", "Mẫn", "Mạnh", "Mậu", "Miên", "Minh", "My", "Nam", "Nga",
    "Ngà", "Ngân", "Ngiêm", "Ngọc", "Nguyệt", "Nháng", "Nhất", "Nhi", "Nhiên",
    "Như", "Nhung", "Oanh", "Phú", "Phụng", "Phương", "Phượng", "Quảng", "Quế",
    "Quý", "Quyên", "Quyền", "Quỳnh", "Sâm", "Sĩ", "Sơn", "Tài", "Tâm", "Tân",
    "Thạch", "Thăng", "Thắng", "Thanh", "Thảo", "Thiên", "Thịnh", "Thọ", "Thư",
    "Thục", "Thúy", "Thủy", "Thy", "Tiên", "Tiến", "Toàn", "Tới", "Trác", "Tràm",
    "Trí", "Trinh", "Trúc", "Trung", "Trường", "Tú", "Tuân", "Tuệ", "Tùng",
    "Tường", "Uyên", "Vân", "Văn", "Vi", "Viễn", "Vinh", "Vĩnh", "Vũ", "Vương",
    "Vy", "Xanh", "Xương", "Ý", "Yến"
]

directions = ['bên trên', 'bên trái', 'bên trên', 'bên dưới', 'phía đông', 'phía tây', 'phía nam', 'phía bắc',
              'hướng đông', 'hướng tây', 'hướng nam', 'hướng bắc',
              'bên tay trái', 'bên tay phải'
              ]
if __name__ == "__main__":
    for i in range(20):
        prefix, type, subject, annotation = create_prefix(include_postfix=False)
        # print(prefix + " " + create_middle('bật thiết bị', include_postfix=True) + ' ' + create_postfix(
        #     'đóng thiết bị',
        # #     subject, type), type)
        # middle, label = create_middle('bật thiết bị', include_postfix=True)
        # post, label_post, type = create_postfix('tắt thiết bị', subject=subject, type=type)
        # print(prefix + " " + middle + " " + post, annotation + " " + label + " " + label_post, type)
        print(prefix, annotation)
