from src.GoogleTrendsScraper import GoogleTrendsScraper
import os
import datetime
import time
import json
import pandas as pd
import numpy as np
import random
from playwright.sync_api import sync_playwright
import traceback

# 游戏关键词列表
def get_game_keywords():
    """
    从主流在线H5游戏网站获取热门游戏关键词
    """
    print("开始获取游戏关键词...")
    
    # 如果已经有缓存的关键词，直接加载
    if os.path.exists('keyword_suggestions.json'):
        with open('keyword_suggestions.json', 'r', encoding='utf-8') as f:
            print("从缓存加载关键词...")
            return json.load(f)
    
    # 否则从网站抓取关键词
    keywords = []
    
    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=False)
            page = browser.new_page()
            
            # 访问Y8游戏网站
            print("访问Y8游戏网站...")
            page.goto('https://www.y8.com/tags/popular', timeout=60000)
            time.sleep(3)
            
            # 获取热门游戏名称
            game_elements = page.query_selector_all('.item-game .item-title')
            for element in game_elements[:20]:  # 获取前20个游戏
                game_name = element.inner_text().strip()
                if game_name and len(game_name) > 2:
                    keywords.append(game_name)
            
            # 访问Poki游戏网站
            print("访问Poki游戏网站...")
            page.goto('https://poki.com/en/popular', timeout=60000)
            time.sleep(3)
            
            # 获取热门游戏名称
            game_elements = page.query_selector_all('.game-card-name')
            for element in game_elements[:20]:  # 获取前20个游戏
                game_name = element.inner_text().strip()
                if game_name and len(game_name) > 2:
                    keywords.append(game_name)
            
            # 访问Crazy Games网站
            print("访问Crazy Games网站...")
            page.goto('https://www.crazygames.com/t/popular', timeout=60000)
            time.sleep(3)
            
            # 获取热门游戏名称
            game_elements = page.query_selector_all('.game__title')
            for element in game_elements[:20]:  # 获取前20个游戏
                game_name = element.inner_text().strip()
                if game_name and len(game_name) > 2:
                    keywords.append(game_name)
            
            browser.close()
    
    except Exception as e:
        print(f"抓取游戏关键词时出错: {str(e)}")
        # 如果抓取失败，使用一些预定义的热门游戏关键词
        keywords = [
            "Minecraft", "Roblox", "Fortnite", "Among Us", "Fall Guys",
            "Subway Surfers", "Wordle", "Candy Crush", "Clash Royale", "PUBG Mobile",
            "Genshin Impact", "Brawl Stars", "Call of Duty Mobile", "Gartic Phone",
            "Stumble Guys", "Slither.io", "Agar.io", "Wormate.io", "Krunker.io",
            "Cookie Clicker", "Retro Bowl", "Slope", "Eggy Car", "Flappy Bird",
            "Temple Run", "Angry Birds", "Cut the Rope", "Fruit Ninja", "Crossy Road",
            "Geometry Dash", "Happy Wheels", "Talking Tom", "Pou", "Hill Climb Racing"
        ]
    
    # 去重并保存关键词
    keywords = list(set(keywords))
    print(f"获取到 {len(keywords)} 个游戏关键词")
    
    # 保存关键词到文件
    with open('keyword_suggestions.json', 'w', encoding='utf-8') as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)
    
    return keywords

# 手动抓取单个关键词的趋势数据
def get_keyword_trend(gts, keyword, reference_keyword, start_date, end_date):
    """
    手动抓取单个关键词的趋势数据
    """
    print(f"\n开始抓取关键词 '{keyword}' 的趋势数据...")
    
    try:
        # 创建URL
        url = gts.create_url([reference_keyword, keyword], 
                            datetime.datetime.strptime(start_date, '%Y-%m-%d'),
                            datetime.datetime.strptime(end_date, '%Y-%m-%d'),
                            'US')
        
        # 获取数据
        data, frequency = gts.get_data(url)
        
        if data.empty:
            print(f"关键词 '{keyword}' 的趋势数据为空")
            return None
            
        # 检查数据列
        print(f"数据列: {data.columns.tolist()}")
        
        # 尝试提取关键词数据
        keyword_data = None
        reference_data = None
        
        # 查找包含关键词的列
        for col in data.columns:
            if keyword.lower() in col.lower():
                keyword_data = data[col]
                print(f"找到关键词列: {col}")
            elif reference_keyword.lower() in col.lower():
                reference_data = data[col]
                print(f"找到参考词列: {col}")
        
        # 如果没有找到列，尝试直接使用列
        if keyword_data is None and len(data.columns) >= 2:
            keyword_data = data.iloc[:, 1]  # 假设第二列是关键词数据
            print(f"使用第二列作为关键词数据: {data.columns[1]}")
        
        if reference_data is None and len(data.columns) >= 1:
            reference_data = data.iloc[:, 0]  # 假设第一列是参考词数据
            print(f"使用第一列作为参考词数据: {data.columns[0]}")
        
        # 如果仍然没有找到数据，返回None
        if keyword_data is None or reference_data is None:
            print(f"无法找到关键词 '{keyword}' 或参考词 '{reference_keyword}' 的数据列")
            return None
        
        # 创建结果数据框
        result_df = pd.DataFrame()
        
        # 添加日期列
        if isinstance(data.index, pd.DatetimeIndex):
            result_df['date'] = data.index
        else:
            # 如果索引不是日期，创建一个日期范围
            date_range = pd.date_range(start=start_date, end=end_date, periods=len(data))
            result_df['date'] = date_range
        
        # 添加其他列
        result_df['keyword'] = keyword
        
        # 处理数据中的"<1"值
        def clean_value(val):
            if isinstance(val, str):
                if val == '<1':
                    return 0.5  # 将"<1"替换为0.5
                try:
                    return float(val)
                except:
                    return 0
            return val
        
        # 清理并转换数据
        keyword_values = [clean_value(val) for val in keyword_data.values]
        reference_values = [clean_value(val) for val in reference_data.values]
        
        result_df['trend_value'] = keyword_values
        result_df['reference_value'] = reference_values
        
        # 计算估计的每日搜索量 (参考词每天约600次搜索)
        # 避免除以零
        result_df['estimated_daily_searches'] = [
            (kw / ref) * 600 if ref > 0 else 0
            for kw, ref in zip(result_df['trend_value'], result_df['reference_value'])
        ]
        
        print(f"关键词 '{keyword}' 的趋势数据抓取成功")
        return result_df
        
    except Exception as e:
        print(f"抓取关键词 '{keyword}' 时出错: {str(e)}")
        traceback.print_exc()
        return None

# 主函数
def main():
    # 获取当前日期和7天前的日期（使用更短的时间范围）
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    
    print(f"抓取时间范围: {start_date} 到 {end_date}")
    
    # 获取游戏关键词
    game_keywords = get_game_keywords()
    
    # 只选择3个关键词进行抓取（减少数量以提高成功率）
    selected_keywords = random.sample(game_keywords, min(3, len(game_keywords)))
    print(f"选择的关键词: {selected_keywords}")
    
    # 添加流量参考词
    reference_keyword = "conversational"
    
    # 初始化结果数据框
    results_df = pd.DataFrame()
    
    try:
        # 初始化抓取器
        print("初始化GoogleTrendsScraper...")
        gts = GoogleTrendsScraper(sleep=15, headless=False, debug=True)
        
        # 为每个关键词抓取趋势数据
        for keyword in selected_keywords:
            # 使用手动抓取方法
            keyword_df = get_keyword_trend(gts, keyword, reference_keyword, start_date, end_date)
            
            if keyword_df is not None:
                # 添加到结果数据框
                results_df = pd.concat([results_df, keyword_df])
                print(f"已添加关键词 '{keyword}' 的数据，当前结果数据框大小: {len(results_df)}")
            
            # 等待一段时间，避免请求过于频繁
            time.sleep(random.uniform(10, 20))
        
        # 保存结果
        print(f"最终结果数据框大小: {len(results_df)}")
        print(f"结果数据框是否为空: {results_df.empty}")
        
        if not results_df.empty:
            # 创建结果目录
            os.makedirs('game_trends_results', exist_ok=True)
            
            # 打印结果数据框的前几行
            print("\n结果数据框预览:")
            print(results_df.head())
            
            # 保存详细结果
            results_file = f'game_trends_results/game_trends_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            results_df.to_csv(results_file, index=False)
            print(f"详细结果已保存到 {results_file}")
            
            # 创建摘要报告
            try:
                summary_df = results_df.groupby('keyword').agg({
                    'trend_value': 'mean',
                    'estimated_daily_searches': ['mean', 'max', 'min']
                })
                
                # 重置多级列名
                summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
                summary_df = summary_df.reset_index()
                
                summary_file = f'game_trends_results/game_trends_summary_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                summary_df.to_csv(summary_file, index=False)
                print(f"摘要报告已保存到 {summary_file}")
            except Exception as summary_error:
                print(f"创建摘要报告时出错: {str(summary_error)}")
                print(f"错误详情: {traceback.format_exc()}")
        else:
            print("没有成功抓取任何数据")
            
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
    
    print("程序执行完毕")

if __name__ == "__main__":
    main() 
