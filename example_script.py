from src.GoogleTrendsScraper import GoogleTrendsScraper
import os
import datetime
import time

# 获取当前日期和7天前的日期（使用更短的时间范围）
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')

print(f"抓取时间范围: {start_date} 到 {end_date}")

# 初始化抓取器 - 使用有界面的浏览器模式
try:
    print("初始化GoogleTrendsScraper...")
    # 设置headless=False以显示浏览器窗口
    gts = GoogleTrendsScraper(sleep=15, headless=False, debug=True)
    
    # 获取趋势数据 - 使用简单的关键词
    print("开始抓取数据...")
    data = gts.get_trends('python', start_date, end_date, 'US')
    
    print("\n抓取结果:")
    print(data)
    
    # 保存结果到CSV
    data.to_csv('python_trends.csv')
    print("结果已保存到 python_trends.csv")
    
except Exception as e:
    print(f"抓取过程中出错: {str(e)}")
finally:
    # 清理资源
    try:
        print("清理资源...")
        if 'gts' in locals():
            del gts
    except Exception as cleanup_error:
        print(f"清理资源时出错: {str(cleanup_error)}")

print("程序执行完毕，请查看截图了解详细过程")

# 注意：首次运行前需要安装playwright浏览器
# 运行以下命令：
# python3 -m playwright install chromium
