                        for item in group_data['group']:
                            price_vs_min = ((item['price'] - group_data['min_price']) /
                                            group_data['min_price'] * 100) if group_data['min_price'] > 0 else 0

                            if item['price'] == group_data['min_price']:
                                marker = "✓ ЛУЧШАЯ ЦЕНА"
                            elif price_vs_min > 50:
                                marker = "⚠️ ПЕРЕПЛАТА >50%"
                            elif price_vs_min > 30:
                                marker = "⚠️ ПЕРЕПЛАТА >30%"
                            else:
                                marker = ""

                            print(f"│")
                            print(f"│ {item['supplier'][:40]:<40}")
                            print(f"│ {item['product'][:70]}")
                            print(f"│ Цена: {item['price']:>15,.0f} eur  (+{price_vs_min:>5.1f}%) {marker}")

                        print("└" + "─" * 78 + "┘")

                    if len(critical_groups) > 5:
                        print(f"\n... и ещё {len(critical_groups) - 5} групп с расхождениями")

                    # Создание отчета по расхождениям
                    print("\n" + "="*80)
                    print("СОЗДАНИЕ ДЕТАЛЬНОГО ОТЧЕТА...")
                    print("="*80)

                    print("[DEBUG] Вызываем find_price_discrepancies...")
                    discrepancies_df = find_price_discrepancies(
                        converted_df,
                        threshold=0.85,
                        method='combined',
                        price_diff_threshold=30.0
                    )

                    if not discrepancies_df.empty:
                        # Сохраняем в Excel
                        output_file = 'price_discrepancies_report.xlsx'
                        discrepancies_df.to_excel(output_file, index=False)
                        print(f"\n✓ Отчет сохранен: {output_file}")

                        # Статистика по отчету
                        print("\nСТАТИСТИКА РАСХОЖДЕНИЙ:")
                        print(f"  • Всего записей в отчете: {len(discrepancies_df)}")
                        print(f"  • Уникальных групп: {discrepancies_df['Группа'].nunique()}")
                        print(f"  • Средняя разница: {discrepancies_df['Разница_%'].mean():.1f}%")
                        print(f"  • Максимальная разница: {discrepancies_df['Разница_%'].max():.1f}%")
                else:
                    print("\n✓ Критических расхождений в ценах не обнаружено")
                    print("   (Товары с похожими названиями найдены, но разница цен <30%)")
            else:
                print("\n! Не найдено групп сопоставимых товаров для анализа")
                print("   Причины:")
                print("   • Слишком мало данных (демо-версия)")
                print("   • Все товары уникальны")
                print("   • Нет товаров от разных поставщиков с похожими названиями")
                print("\n   💡 РЕШЕНИЕ: Загрузите реальные данные из вашей БД SQLite")
        except Exception as e:
            print(f"\n✗ Ошибка при анализе: {e}")
            print("  Проверьте наличие необходимых колонок в датафрейме")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*80)
        print("⚠️ БЛОК 1 ПРОПУЩЕН - product_matcher не загружен")
        print("="*80)
        print("\nЧтобы активировать интеллектуальное сопоставление:")
        print("1. Создайте папку utils/ в директории проекта")
        print("2. Поместите туда файл product_matcher.py")
        print("3. Создайте utils/__init__.py с импортами")
        print("4. Перезапустите скрипт")

    # БЛОК 2. СРАВНЕНИЕ ЦЦЕН МЕЖДУ ПОСТАВЩИКАМИ
    print("\n" + "="*80)
    print("БЛОК 2: БАЗОВОЕ СРАВНЕНИЕ ЦЕН")
    print("="*80)

    converted_df['product_type'] = converted_df['product_name'].str.split().str[0].str.capitalize()

    # Анализ цен по тиам товаров от разных поставщиков
    print("\nСРАВНЕНИЕ ЦЕН ПО ТИПАМ ТОВАРОВ:")
    print("-" * 80)

    comparison_results = []
    for product_type in converted_df['product_type'].unique():
        subset = converted_df[converted_df['product_type'] == product_type]

        if len(subset) >= 2 and subset['counterparty_name'].nunique() >= 2:
            price_by_supplier = subset.groupby('counterparty_name')['unit_price_eur'].agg(['mean', 'count'])

            if len(price_by_supplier) >= 2:
                max_price = price_by_supplier['mean'].max()
                min_price = price_by_supplier['mean'].min()
                price_diff_percent = ((max_price - min_price) / min_price * 100) if min_price > 0 else 0

                comparison_results.append({
                    'Товар': product_type,
                    'Поставщиков': len(price_by_supplier),
                    'Мин_цена_EUR': f"{min_price:,.0f}",
                    'Макс_цена_EUR': f"{max_price:,.0f}",
                    'Разница_%': f"{price_diff_percent:.1f}%"
                })

    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        print(comparison_df.to_string(index=False))

        # Выделяем товары с большой разницей в ценах
        print("\n⚠️ КРИТИЧЕСКИЕ РАСХОЖДЕНИЯ (разница >30%):")
        for result in comparison_results:
            diff = float(result['Разница_%'].rstrip('%'))
            if diff > 30:
                print(f"   • {result['Товар']}: разница {result['Разница_%']} между поставщиками")
    else:
        print("Недостаточно данных для сравнения")

# ================= БЛОК 3. АНАЛИЗ КОНКУРЕНТНОСТИ ============
    print("\n" + "="*80)
    print("БЛОК 3: АНАЛИЗ КОНКУРЕНТНОСТИ ЗАКУПОК")
    print("="*80)

    # Подсчет количества поставщиков по Лотам
    lot_competition = converted_df.groupby('lot_number').agg({
        'counterparty_name': 'nunique',
        'total_amount_eur': 'first',
        'discipline': 'first'
    }).rename(columns={'counterparty_name': 'suppliers_count'})

    # Статистика по конкурентности
    print("\nРАСПРЕДЕЛЕНИЕ ЛОТОВ ПО КОЛИЧЕСТВУ ПОСТАВЩИКОВ:")
    competition_stats = lot_competition['suppliers_count'].value_counts().sort_index()
    for count, freq in competition_stats.items():
        percentage = (freq / len(lot_competition) * 100)
        print(f"   {count} поставщик(ов): {freq} лотов ({percentage:.1f}%)")

    # Лоты с единственным поставщиком (высокий риск)
    single_supplier_lots = lot_competition[lot_competition['suppliers_count'] == 1]
    print(f"\n⚠️ ЛОТЫ С ЕДИНСТВЕННЫМ ПОСТАВЩИКОМ: {len(single_supplier_lots)}")

    if len(single_supplier_lots) > 0:
        print(f"   Общая сумма: {single_supplier_lots['total_amount_eur'].sum()/1e6:.2f}  EUR")
        print("\n   Топ-5 самых крупных:")
        top_single = single_supplier_lots.nlargest(5, 'total_amount_eur')
        for idx, row in top_single.iterrows():
            print(f"   • Лот {idx}: {row['total_amount_eur']/1e6:.2f}  EUR ({row['discipline']})")

    # ============ БЛОК 4. АНАЛИЗ ПАТТЕРНОВ ПОБЕДИТЕЛЕЙ =========
    print("\n" + "="*80)
    print("БЛОК 4: АНАЛИЗ ПАТТЕРНОВ ПОБЕДИТЕЛЕЙ")
    print("="*80)

    # Частота побед поставщиков
    supplier_wins = converted_df.groupby('counterparty_name').agg({
        'lot_number': 'nunique',
        'total_amount_eur': 'sum',
        'discipline': lambda x: list(x.unique())
    }).rename(columns={'lot_number': 'wins_count'}).sort_values('wins_count', ascending=False)

    print("\nПОСТАВЩИКИ С НАИБОЛЬШИМ КОЛИЧЕСТВОМ ПОБЕД:")
    for idx, row in supplier_wins.head(10).iterrows():
        disciplines = ', '.join(row['discipline'][:3])
        if len(row['discipline']) > 3:
            disciplines += f" (+{len(row['discipline'])-3})"
        print(f"   • {idx}: {row['wins_count']} побед, {row['total_amount_eur']/1e6:.1f} eur")
        print(f"     Дисциплины: {disciplines}")

    # Проверка на монополизацию дисциплин
    print("\n⚠️ ДОМИНИРОВАНИЕ В ДИСЦИПЛИНАХ:")
    for discipline in df['discipline'].unique():
        discipline_df = df[df['discipline'] == discipline]
        top_supplier = discipline_df.groupby('counterparty_name')['total_amount_eur'].sum().sort_values(ascending=False)

        if len(top_supplier) > 0:
            total_discipline = top_supplier.sum()
            top_share = (top_supplier.iloc[0] / total_discipline * 100) if total_discipline > 0 else 0

            if top_share > 50:
                print(f"   • {discipline}: {top_supplier.index[0]} контролирует {top_share:.1f}% рынка")

    # ======= БЛОК 5: ВРЕМЕННОЙ АНАЛИЗ ========
    print("\n" + "="*80)
    print("БЛОК 4: ВРЕМЕННОЙ АНАЛИЗ")
    print("="*80)

    # Конвертация дат
    converted_df['lot_end_date'] = pd.to_datetime(converted_df['lot_end_date'], errors='coerce')
    converted_df['contract_signing_date'] = pd.to_datetime(converted_df['contract_signing_date'], errors='coerce')

    # Расчет времени от окончания лота до подписания контракта
    converted_df['days_to_sign'] = (converted_df['contract_signing_date'] - converted_df['lot_end_date']).dt.days

    valid_days = converted_df[converted_df['days_to_sign'].notna() & (converted_df['days_to_sign'] >= 0)]

    if len(valid_days) > 0:
        print("\nСКОРОСТЬ ПОДПИСАНИЯ КОНТРАКТОВ:")
        print(f"   Среднее время: {valid_days['days_to_sign'].mean():.1f} дней")
        print(f"   Медиана: {valid_days['days_to_sign'].median():.1f} дней")
        print(f"   Мин/Макс: {valid_days['days_to_sign'].min():.0f} / {valid_days['days_to_sign'].max():.0f} дней")

        # Подозрительно быстрые подписания
        fast_contracts = valid_days[valid_days['days_to_sign'] == 0]
        if len(fast_contracts) > 0:
            print(f"\n⚠️ Контракты подписаны В ДЕНЬ окончания лота: {len(fast_contracts)}")

    # Анализ по месяцам
    converted_df['month'] = converted_df['lot_end_date'].dt.to_period('M')
    monthly_stats = converted_df.groupby('month').agg({
        'lot_number': 'count',
        'total_amount_eur': 'sum'
    }).rename(columns={'lot_number': 'количество', 'total_amount_eur': 'сумма'})

    if len(monthly_stats) > 0:
        print("\nАКТИВНОСТЬ ПО МЕСЯЦАМ:")
        for month, row in monthly_stats.iterrows():
            print(f"   {month}: {row['количество']} лотов, {row['сумма']/1e6:.1f}  eur")

    # ============ БЛОК 6: АНАЛИЗ СООТВЕТСТВИЯ КОЛИЧЕСТВО/ЦЕНА ========
    print("\n" + "="*80)
    print("БЛОК 5: ПРОВЕРКА МАТЕМАТИЧЕСКОЙ КОРРЕКТНОСТИ")
    print("="*80)

    # Проверка формулы: количество * цена_за_единицу = сумма
    converted_df['calculated_amount'] = converted_df['quantity'] * converted_df['unit_price_eur']
    converted_df['amount_discrepancy'] = abs(converted_df['calculated_amount'] - converted_df['total_amount_eur'])
    converted_df['discrepancy_percent'] = (converted_df['amount_discrepancy'] / converted_df['total_amount_eur'] * 100).fillna(0)

    errors = converted_df[converted_df['discrepancy_percent'] > 1]  # Расхождение >1%

    if len(errors) > 0:
        print(f"\n⚠️ ОБНАРУЖЕНЫ МАТЕМАТИЧЕСКИЕ НЕСООТВЕТСТВИЯ: {len(errors)} записей")
        print("\nПримеры:")
        for idx, row in errors.head(5).iterrows():
            print(f"   • Лот {row['lot_number']}: {row['product_name'][:50]}")
            print(f"     Ожидается: {row['calculated_amount']:,.0f} eur")
            print(f"     В контракте: {row['total_amount_eur']:,.0f} eur")
            print(f"     Расхождение: {row['discrepancy_percent']:.2f}%")
    else:
        print("\n✓ Математические расхождения не обнаружены")

    # ================= БЛОК 7: ВИЗУАЛИЗАЦИЯ ===============
    print("\n" + "="*80)
    print("БЛОК 6: СОЗДАНИЕ АНАЛИТИЧЕСКИХ ГРАФИКОВ")
    print("="*80)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # График 1: Распределение по конкурентности
    ax1 = fig.add_subplot(gs[0, 0])
    competition_stats.plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_title('Распределение лотов\nпо числу участников', fontsize=10)
    ax1.set_xlabel('Количество поставщиков')
    ax1.set_ylabel('Число лотов')
    ax1.grid(True, alpha=0.3)

    # График 2: Топ поставщиков по победам
    ax2 = fig.add_subplot(gs[0, 1])
    top_winners = supplier_wins.head(8)['wins_count']
    ax2.barh(range(len(top_winners)), top_winners.values, color='coral')
    ax2.set_yticks(range(len(top_winners)))
    ax2.set_yticklabels([name[:25] for name in top_winners.index], fontsize=8)
    ax2.set_title('Топ-8 победителей конкурсов', fontsize=10)
    ax2.set_xlabel('Количество побед')
    ax2.grid(True, alpha=0.3, axis='x')

    # График 3: Распределение сумм контрактов
    ax3 = fig.add_subplot(gs[0, 2])
    converted_df['total_amount_eur'].hist(bins=20, ax=ax3, color='green', alpha=0.7)
    ax3.set_title('Распределение сумм контрактов', fontsize=10)
    ax3.set_xlabel('Сумма контракта (EUR)')
    ax3.set_ylabel('Частота')
    ax3.grid(True, alpha=0.3)

    # График 4: Цены по типам товаров
    ax4 = fig.add_subplot(gs[1, 0])
    price_by_type = converted_df.groupby('product_type')['unit_price_eur'].mean().sort_values(ascending=False).head(8)
    ax4.barh(range(len(price_by_type)), price_by_type.values, color='purple', alpha=0.7)
    ax4.set_yticks(range(len(price_by_type)))
    ax4.set_yticklabels(price_by_type.index, fontsize=8)
    ax4.set_title('Средняя цена по типам товаров', fontsize=10)
    ax4.set_xlabel('Цена за единицу (eur)')
    ax4.grid(True, alpha=0.3, axis='x')

    # График 5: Условия оплаты
    ax5 = fig.add_subplot(gs[1, 1])
    payment_dist = converted_df.groupby('payment_conditions')['total_amount_eur'].sum().sort_values(ascending=False).head(5)
    ax5.pie(payment_dist.values, labels=[p[:20]+'...' if len(p)>20 else p for p in payment_dist.index],
            autopct='%1.1f%%', startangle=90)
    ax5.set_title('Распределение по условиям оплаты', fontsize=10)

    # График 6: Сроки поставки по дисциплинам
    ax6 = fig.add_subplot(gs[1, 2])
    delivery_by_discipline = converted_df.groupby('discipline')['delivery_time_days'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').mean()
    ).sort_values(ascending=False)
    ax6.barh(range(len(delivery_by_discipline)), delivery_by_discipline.values, color='orange', alpha=0.7)
    ax6.set_yticks(range(len(delivery_by_discipline)))
    ax6.set_yticklabels([d[:20] for d in delivery_by_discipline.index], fontsize=8)
    ax6.set_title('Средние сроки поставки', fontsize=10)
    ax6.set_xlabel('Дней')
    ax6.grid(True, alpha=0.3, axis='x')

    # График 7: Динамика по месяцам
    ax7 = fig.add_subplot(gs[2, :2])
    if len(monthly_stats) > 0:
        monthly_stats['количество'].plot(kind='line', marker='o', ax=ax7, color='blue', label='Количество лотов')
        ax7_twin = ax7.twinx()
        monthly_stats['сумма'].plot(kind='line', marker='s', ax=ax7_twin, color='red', label='Сумма (eur)')
        ax7.set_title('Динамика закупок по месяцам', fontsize=10)
        ax7.set_xlabel('Месяц')
        ax7.set_ylabel('Количество лотов', color='blue')
        ax7_twin.set_ylabel('Сумма (EUR)', color='red')
        ax7.legend(loc='upper left')
        ax7_twin.legend(loc='upper right')
        ax7.grid(True, alpha=0.3)

    # График 8: Сравнение поставщиков по дисциплинам
    ax8 = fig.add_subplot(gs[2, 2])
    discipline_supplier = converted_df.groupby(['discipline', 'counterparty_name'])['total_amount_eur'].sum().unstack(fill_value=0)
    if not discipline_supplier.empty:
        discipline_supplier.iloc[:, :5].plot(kind='bar', stacked=True, ax=ax8, legend=False)
        ax8.set_title('Структура поставщиков\nпо дисциплинам', fontsize=10)
        ax8.set_xlabel('Дисциплина')
        ax8.set_ylabel('Сумма контрактов (EUR)')
        plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.savefig('advanced_procurement_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Графики сохранены в 'advanced_procurement_analysis.png'")

    # ================ ИТОГОВЫЙ ОТЧЕТ ============
    print("\n" + "="*80)
    print("ИТОГОВЫЙ ОТЧЕТ - КЛЮЧЕВЫЕ ИНДИКАТОРЫ РИСКА")
    print("="*80)

    total_lots = converted_df['lot_number'].nunique()
    total_amount = converted_df['total_amount_eur'].sum()
    avg_competition = lot_competition['suppliers_count'].mean()

    risk_score = 0
    risk_factors = []

    # Фактор 1: Низкая конкурентность
    if len(single_supplier_lots) / total_lots > 0.3:
        risk_score += 3
        risk_factors.append(f"Высокая доля лотов с 1 поставщиком ({len(single_supplier_lots)/total_lots*100:.1f}%)")
    elif len(single_supplier_lots) / total_lots > 0.1:
        risk_score += 1
        risk_factors.append(f"Умеренная доля лотов с 1 поставщиком ({len(single_supplier_lots)/total_lots*100:.1f}%)")

    # Фактор 2: Концентрация у победителей
    top3_share = supplier_wins.head(3)['total_amount_eur'].sum() / total_amount
    if top3_share > 0.6:
        risk_score += 3
        risk_factors.append(f"Высокая концентрация у топ-3 поставщиков ({top3_share*100:.1f}%)")
    elif top3_share > 0.4:
        risk_score += 1
        risk_factors.append(f"Умеренная концентрация у топ-3 поставщиков ({top3_share*100:.1f}%)")

    # Фактор 3: Рискованные условия оплаты
    prepayment_100 = converted_df[converted_df['payment_conditions'].str.contains('100%', na=False)]
    if len(prepayment_100) / len(converted_df) > 0.5:
        risk_score += 2
        risk_factors.append(f"Много контрактов со 100% предоплатой ({len(prepayment_100)/len(converted_df)*100:.1f}%)")

    # Фактор 4: Математические ошибки
    if len(errors) > 0:
        risk_score += 2
        risk_factors.append(f"Обнаружены математические несоответствия ({len(errors)} позиций)")

    # Фактор 5: Быстрые подписания
    if len(valid_days) > 0:
        fast_ratio = len(converted_df[converted_df['days_to_sign'] == 0]) / len(valid_days)
        if fast_ratio > 0.3:
            risk_score += 1
            risk_factors.append(f"Много контрактов подписано в день окончания лота ({fast_ratio*100:.1f}%)")

    print(f"""
    ┌{'─'*78}┐
    │ ОБЩАЯ СТАТИСТИКА                                                            │
    ├{'─'*78}┤
    │ • Всего лотов: {total_lots:<60} │
    │ • Уникальных поставщиков: {converted_df['counterparty_name'].nunique():<50} │
    │ • Общая сумма контрактов: {total_amount/1e9:.2f} млрд eur{' '*36} │
    │ • Средняя конкурентность: {avg_competition:.2f} поставщика на лот{' '*31} │
    └{'─'*78}┘

    ┌{'─'*78}┐
    │ ИНДИКАТОРЫ РИСКА                                                            │
    ├{'─'*78}┤
    │ Общий балл риска: {risk_score}/10                                                      │
    │                                                                              │""")

    if risk_score == 0:
        print("│ ✓ Критических рисков не обнаружено                                       │")
    elif risk_score <= 3:
        print("│ ⚠ Низкий уровень риска                                                   │")
    elif risk_score <= 6:
        print("│ ⚠⚠ Средний уровень риска - требуется внимание                           │")
    else:
        print("│ ⚠⚠⚠ ВЫСОКИЙ УРОВЕНЬ РИСКА - требуется детальная проверка               │")

    print("│                                                                              │")
    print("│ Обнаруженные факторы риска:                                                  │")

    if risk_factors:
        for i, factor in enumerate(risk_factors, 1):
            print(f"│ {i}. {factor:<73}│")
    else:
        print("│ • Факторы риска не обнаружены                                            │")

    print(f"└{'─'*78}┘")

    # Детальные рекомендации
    print(f"""
    ┌{'─'*78}┐
    │ РЕКОМЕНДАЦИИ ДЛЯ КОНТРОЛЬНО-РЕВИЗИОННОГО ДЕПАРТАМЕНТА                        │
    ├{'─'*78}┤""")

    recommendations = []

    if len(single_supplier_lots) > 0:
        recommendations.append(
            "1. КОНКУРЕНТНОСТЬ:\n"
            f"   • Проверить причины участия 1 поставщика в {len(single_supplier_lots)} лотах\n"
            "   • Оценить барьеры входа для других участников\n"
            "   • Рассмотреть разукрупнение лотов"
        )

    if len(comparison_results) > 0:
        high_diff = [r for r in comparison_results if float(r['Разница_%'].rstrip('%')) > 30]
        if high_diff:
            recommendations.append(
                "2. ЦЕНООБРАЗОВАНИЕ:\n"
                f"   • Проверить {len(high_diff)} товаров с расхождением цен >30%\n"
                "   • Запросить обоснование цен у поставщиков\n"
                "   • Провести независимую оценку рыночных цен"
            )

    if top3_share > 0.5:
        recommendations.append(
            "3. ДИВЕРСИФИКАЦИЯ:\n"
            f"   • Топ-3 поставщика контролируют {top3_share*100:.1f}% рынка\n"
            "   • Изучить возможности привлечения новых поставщиков\n"
            "   • Оценить риски зависимости от ключевых контрагентов"
        )

    if len(errors) > 0:
        recommendations.append(
            "4. МАТЕМАТИЧЕСКАЯ ПРОВЕРКА:\n"
            f"   • Исправить {len(errors)} записей с расхождением количество×цена\n"
            "   • Внедрить автоматическую валидацию при вводе данных\n"
            "   • Проверить все контракты за период"
        )

    if len(prepayment_100) > len(converted_df) * 0.3:
        recommendations.append(
            "5. УСЛОВИЯ ОПЛАТЫ:\n"
            f"   • {len(prepayment_100)} контрактов со 100% предоплатой\n"
            "   • Оценить финансовые риски для компании\n"
            "   • Рассмотреть альтернативные схемы оплаты"
        )

    if len(valid_days) > 0 and len(converted_df[converted_df['days_to_sign'] == 0]) > 0:
        recommendations.append(
            "6. ПРОЦЕДУРНЫЕ ВОПРОСЫ:\n"
            f"   • {len(converted_df[converted_df['days_to_sign'] == 0])} контрактов подписано немедленно\n"
            "   • Проверить соблюдение сроков на обжалование\n"
            "   • Оценить достаточность времени для оценки заявок"
        )

    if recommendations:
        for rec in recommendations:
            for line in rec.split('\n'):
                print(f"│ {line:<76} │")
    else:
        print("│ • Критических замечаний нет - продолжить плановый мониторинг             │")

    print(f"└{'─'*78}┘")

    print(f"""
    ┌{'─'*78}┐
    │ ПРИОРИТЕТЫ ПРОВЕРКИ                                                          │
    ├{'─'*78}┤
    │ ВЫСОКИЙ ПРИОРИТЕТ (проверить в первую очередь):                              │
    │ • Лоты с единственным поставщиком на суммы >{converted_df['total_amount_eur'].quantile(0.75)/1e6:.1f} млн eur{' '*20}│
    │ • Товары с ценовыми расхождениями >50%                                       │
    │ • Контракты с математическими несоответствиями                               │
    │                                                                              │
    │ СРЕДНИЙ ПРИОРИТЕТ:                                                           │
    │ • Контракты со 100% предоплатой                                              │
    │ • Быстрые подписания (0-1 день)                                              │
    │ • Повторяющиеся победители в одной дисциплине                                │
    │                                                                              │
    │ НИЗКИЙ ПРИОРИТЕТ (плановый мониторинг):                                      │
    │ • Мелкие закупки (<1 млн eur)                                                │
    │ • Контракты с конкурентными условиями                                        │
    │ • Товары со стандартными рыночными ценами                                    │
    └{'─'*78}┘

    ┌{'─'*78}┐
    │ ДОПОЛНИТЕЛЬНЫЕ ИНСТРУМЕНТЫ АНАЛИЗА                                           │
    ├{'─'*78}┤
    │ Для углубленной проверки используйте:                                        │
    │                                                                              │
    │ 1. Модуль 1: Базовый анализ аномалий                                         │
    │    • Выявление ценовых выбросов                                              │
    │    • Анализ поставщиков                                                      │
    │    • Общая визуализация                                                      │
    │                                                                              │
    │ 2. Модуль 2 (текущий): Расширенный анализ                                    │
    │    • Сравнение между поставщиками                                            │
    │    • Анализ конкурентности                                                   │
    │    • Временные паттерны                                                      │
    │    • Проверка корректности данных                                            │
    │                                                                              │
    │ 3. Экспорт результатов для отчетности:                                       │
    │    • График: advanced_procurement_analysis.png                               │
    │    • Можно добавить экспорт в Excel/PDF                                      │
    └{'─'*78}┘
    """)

    print("="*80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("="*80)