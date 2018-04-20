# coding:utf-8
class _const:
    class ConstError(TypeError):
        pass
    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value


const = _const()

const.VDS_HEAD = '/?report_form=1&dnode=Freeway&content=elv&export=xls' \
                 '&eqpo=&tag=&st_cd=on&st_ch=on&st_ff=on&st_hv=on&st_ml=on&st_fr=on&st_or=on&'

const.DATA_HEAD = '/?report_form=1&dnode=VDS&content=loops&tab=det_timeseries&export=xls&tod=all' \
                  '&tod_from=0&tod_to=0&dow_0=on&dow_1=on&dow_2=on&dow_3=on&dow_4=on&dow_5=on' \
                  '&dow_6=on&holidays=on&agg=on&lane1=on&lane2=on&lane3=on&lane4=on&lane5=on&'

const.DATA_FROM = {
    'redirect': '',
    'username': 'zydarchen@outlook.com',
    'password': 'treep9:rQ',
    'login': 'Login',
}

# 美国十大节假日
const.VACATION = ['2016-01-01', '2016-01-18', '2016-02-15',
                  '2016-05-30', '2016-07-04', '2016-09-05',
                  '2016-10-10', '2016-11-11', '2016-11-24',
                  '2016-11-25', '2016-12-23', '2016-12-26',
                  ]
