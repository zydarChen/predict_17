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
const.HEAD = '/?report_form=1&dnode=VDS&content=loops&tab=det_timeseries&export=xls&tod=all&tod_from=0&tod_to=0&dow_0' \
             '=on&dow_1=on&dow_2=on&dow_3=on&dow_4=on&dow_5=on&dow_6=on&holidays=on&q=flow&q2=&gn=5min&agg=on&lane1' \
             '=on&lane2=on&lane3=on '